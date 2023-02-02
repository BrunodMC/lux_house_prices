import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
import os
import pandas as pd
import random

from utils import _setup_directory


# step 1: scrape website for properties for sale
# step 2: convert links to properties advertised into database entries with the relevant info

################################################
### Functions to find all relevant articles
def extract_athomelu_entries():
    """Scrapes athome.lu, collecting the URL to every single property advertised in Luxembourg and writing them to a file.
        (took 21 minutes to run in my test)"""

    # quick setup
    _setup_directory()

    st_time = time.time()
    BASE_URL = "https://www.athome.lu"

    # get HTML from site
    first_URL = BASE_URL + '/en/buy'
    site = requests.get(first_URL)
    site_soup = BeautifulSoup(site.content, "html.parser")

    # find total number of results
    total_results = site_soup.find_all('header', class_='block-alert')[0]
    total_results = total_results.find_all('h2')[0].text.split(' ')[0]
    total_results = int(total_results.replace(',', ''))
    print(f"Total search results: {total_results}")
    
    # find total number of pages of search results
    num_result_pages = int(site_soup.find_all('a', class_='page last')[0].text)
    print(f"Total number of search result pages: {num_result_pages}")

    # save all article URLs to a txt file
    current_filepath = os.path.dirname(os.path.abspath(__file__))
    timestr = datetime.now().strftime("%Y%m%d%H%M%S")
    filepath = current_filepath + f"/extracted_URLs/URLs_{timestr}.txt"
    with open(filepath, 'w+') as file:
        # loop through all results pages
        for i in range(1,num_result_pages+1):
            PAGE_URL = BASE_URL + f"/en/buy?page={i}" 
            page = requests.get(PAGE_URL)
            page_soup = BeautifulSoup(page.content, 'html.parser')
            # find all articles displayed in the current page
            articles = page_soup.find_all('article')
            # extract the useful info for the given article
            for article in articles:
                # first of all, check if property is in luxembourg
                if _not_in_lux(article): continue
                # check if "<p>: class=childrenInfos" exists
                collective = bool(article.find_all('p', class_='childrenInfos'))
                if collective:
                    href_list = _collective_article(article, BASE_URL)
                    url_list = [BASE_URL + href + '\n' for href in href_list]
                    file.writelines(url_list)
                else:
                    PROPERTY_URL = BASE_URL + _individual_article(article) + '\n'
                    file.write(PROPERTY_URL)

    file.close()
    # remove duplicate URLs
    with open(filepath, 'r') as read:
        all_lines = set(read.readlines())
    read.close()

    with open(filepath, 'w+') as rewrite:
        for line in all_lines:
            rewrite.write(line)
    rewrite.close()

    et_time = time.time()
    print(f"Successfully wrote all relevant URLs to file with path '{filepath}'.")
    print(f"This process took {round(et_time - st_time, 2)} seconds.")
    return

def _not_in_lux(article):
    """Returns True if the property is not in Luxembourg."""

    locality = article.find_all('span', itemprop='addressLocality')[0].text
    check = locality.split('(')[-1]
    if 'FR)' in check or 'DE)' in check or 'BE)' in check:
        return True
    return False

def _individual_article(article):
    """Returns a string of the href of the property."""

    return article.find_all('link', itemprop='url')[0]['href']

def _collective_article(article, BASE_URL):
    """Returns a list of href strings corresponding to each property included in the collective."""

    # Extract collective property page's URLs to each individual property
    col_prop_page_url = BASE_URL + article.find_all('link', itemprop='url')[0]['href']
    collective_page = requests.get(col_prop_page_url)
    collective_soup = BeautifulSoup(collective_page.content, 'html.parser')
    property_divs = collective_soup.find_all('div', class_='residence-informations-content')
    
    href_list = []
    for property in property_divs:
        href = property.find_all('a')[0]['href']
        href_list.append(href)
    
    return href_list


################################################
### Functions to extract the actual data from the relevant articles

def get_data():
    """Collects the data for every property in the most recent collection of URLs and saves it to CSV."""

    # quick setup
    _setup_directory()

    st_time = time.time()
    # find the most up to date set of URLs
    current_filepath = os.path.dirname(os.path.abspath(__file__))
    URLs_filepath = current_filepath + '/extracted_URLs/'
    list_of_files = os.listdir(URLs_filepath)
    # raise error if there are no files in the directory
    if not len(list_of_files):
        raise Exception(f'No files exist in {URLs_filepath}')

    file_timestamps = [int(x.split('_')[-1][:-4]) for x in list_of_files]
    target = str(max(file_timestamps))
    target_filepath = URLs_filepath + [file for file in list_of_files if target in file][0]
    
    # get the relevant information from each advert
    data = []
    with open(target_filepath, 'r') as file:
        for i, url in enumerate(file):
            page = requests.get(url.strip()) 
            # check if ad still exists
            if page.status_code != 200:
                print(f'Something went wrong with url number {i+1}: {url}')
                continue

            page_soup = BeautifulSoup(page.content, 'html.parser')
            # get a couple of specific things
            _sentence_list = page_soup.find_all('h1', class_='KeyInfoBlockStyle__PdpTitle-sc-1o1h56e-2 ilPGib')[0].text.split(' ')
            type_of_property = _sentence_list[0]
            _in_index = _sentence_list.index('in')
            locality = _sentence_list[_in_index+1]

            # get everything in the characteristics block of the page
            try:
                _characteristics_block = page_soup.find_all('section', class_='feature sc-7vp35h-2-section-LayoutTheme__KeyGeneral-hbgJJa hVtovK')[0]
            except:
                print(f"URL number {i+1} might have no info.")
            else:
                characteristics_dict = _scan_characteristics_block(_characteristics_block)

                characteristics_dict['Property Type'] = type_of_property
                characteristics_dict['Locality'] = locality

                data.append(characteristics_dict)
    file.close()

    # turn the whole thing into a dataframe to save it as a CSV for future reference
    df = pd.DataFrame(data)
    csv_path = current_filepath + '/raw_datasets/' + f'data_{target}.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8')

    et_time = time.time()
    print(f"Successfully saved data to CSV file with path '{csv_path}'.")
    print(f"This process took {round(et_time - st_time, 2)} seconds.")

    return

def _scan_characteristics_block(block):
    
    pairs = block.find_all('li', class_='feature-bloc-content-specification-content')
    # scan through each key value pair in the block, logging them in a dictionary
    data = {}
    for pair in pairs:
        key = pair.find_all('div', class_='feature-bloc-content-specification-content-name')[0].text.strip()
        value = pair.find_all('div', class_='feature-bloc-content-specification-content-response')[0].text.strip()
        data[key] = value
    
    return data


################################################
### Other
def _find_characteristics():
    # find the most up to date set of URLs
    current_filepath = os.path.dirname(os.path.abspath(__file__))
    URLs_filepath = current_filepath + '/extracted_URLs/'
    
    target_filepath = URLs_filepath + 'URLs_20230126155023.txt'

    L = []
    with open(target_filepath, 'r') as file:
        for i in range(1000):
            url = file.readline()
            page = requests.get(url.strip())
            page_soup = BeautifulSoup(page.content, 'html.parser')
            try:
                _characteristics_block = page_soup.find_all('section', class_='feature sc-7vp35h-2-section-LayoutTheme__KeyGeneral-hbgJJa hVtovK')[0]
            except:
                print(f"Something went wrong at i = {i}")
            else:
                pairs = _characteristics_block.find_all('li', class_='feature-bloc-content-specification-content')
                for pair in pairs:
                    key = pair.find_all('div', class_='feature-bloc-content-specification-content-name')[0].text.strip()
                    if key not in L: L.append(key)
            
    print(L)
    print(len(L))

def _test():
    url = 'https://www.athome.lu/en/buy/apartment/hesperange/id-7707193.html'
    page = requests.get(url)

    print(page.status_code)

def _gather_subset() -> None:
    current_filepath = os.path.dirname(os.path.abspath(__file__))
    URLs_filepath = current_filepath + '/extracted_URLs/'
    
    source_filepath = URLs_filepath + 'URLs_20230126155023.txt'

    with open(source_filepath, 'r') as f:
        all_lines = list(set(f.readlines()))
    f.close()

    subset = set([all_lines[random.randint(0, len(all_lines)-1)] for _ in range(1000)])
    

    target_filepath = URLs_filepath + 'URLs_20230126156000.txt'
    with open(target_filepath, 'w+') as f:
        for line in subset:
            f.write(line)
    f.close()

    return


if __name__ == '__main__':
    get_data()
    # _find_characteristics()
    # _test()
    # _setup_directory()
    # _gather_subset()
    pass