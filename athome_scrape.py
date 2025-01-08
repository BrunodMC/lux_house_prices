import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
import os
import pandas as pd
import random
import warnings

from utils import _setup_directory, _find_file


# simpler warning formatting
def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return '%s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)
warnings.formatwarning = warning_on_one_line

################################################
### Functions to find all relevant articles
def extract_athomelu_entries():
    """Scrapes athome.lu, collecting the URL to every single property advertised in Luxembourg and writing them to a file.
        (took ~40 minutes to run for 41k alleged results (20k parsed articles and 10k saved URLs))"""

    # quick setup
    _setup_directory()

    st_time = time.time()
    BASE_URL = "https://www.athome.lu"

    # get HTML from site
    first_URL = BASE_URL + '/en/buy'
    site = requests.get(first_URL)
    site_soup = BeautifulSoup(site.content, "html.parser")

    # find total number of results
    total_results = site_soup.find_all('header', class_='block-alert-top')[0]
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
    saved_url_counter = 0
    parsed_article_counter = 0
    printcounter = 200
    with open(filepath, 'w+') as file:
        # use a set to keep track of property ID's and ensure we don't save duplicates
        hashset_property_id = set()
        # loop through all results pages
        for i in range(1,num_result_pages+1):
            page_url = BASE_URL + f"/en/buy?page={i}" 
            page = requests.get(page_url)
            page_soup = BeautifulSoup(page.content, 'html.parser')
            # find all articles displayed in the current page
            articles = page_soup.find_all('article')
            # extract the useful info for the given article
            for article in articles:
                parsed_article_counter += 1
                # first of all, check if property is in luxembourg
                if _not_in_lux(article): continue
                # next, check if property ID is already known (in the hashset)
                href = _individual_article(article)
                prop_id = int(''.join(filter(str.isnumeric, href.split('/')[-1])))
                # check if it is already in the set, if so skip it
                if prop_id in hashset_property_id:
                    continue
                # if not in the set, add it and proceed as normal
                hashset_property_id.add(prop_id)
                # check if "<p>: class=childrenInfos" exists, meaning the property is collective
                collective = bool(article.find_all('p', class_='childrenInfos'))
                if collective:
                    href_list = _collective_article(article, BASE_URL)
                    url_list = [BASE_URL + href + '\n' for href in href_list]
                    file.writelines(url_list)
                    saved_url_counter += len(url_list)
                else:
                    property_url = BASE_URL + _individual_article(article) + '\n'
                    file.write(property_url)
                    saved_url_counter += 1
                
                # print counter every ~200 articles or so
                if (parsed_article_counter >= printcounter):
                    if printcounter == 200:
                        print("Number of Articles parsed (URLs collected)...")
                    print(f"{parsed_article_counter} ({saved_url_counter})")
                    printcounter += 200
                
    # close file
    file.close()
    # print some info
    et_time = time.time()
    print(f"Found {saved_url_counter} relevant URLs, wrote them to file with path '{filepath}'.")
    print(f"This process took {round(et_time - st_time, 2)} seconds ({round((et_time - st_time)/60, 2)} minutes).")
    return

def _not_in_lux(article):
    """Returns True if the property is NOT in Luxembourg."""

    country_span = article.find('span', {'class':'property-card-immotype-location-country'})
    if country_span: # if not None, country was specified, which only happens if outside of Lux
        return True
    return False

def _individual_article(article):
    """Returns a string of the href of the property."""

    return article.find('link', itemprop='url')['href']

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
    """Collects the data for every property in the most recent collection of URLs and saves it to CSV.
        (took ~20 minutes in my test)"""

    # quick setup
    _setup_directory()

    st_time = time.time()
    batch_st_time = st_time
    # find the most up to date set of URLs
    target_filepath, target_timestamp = _find_file('extracted_URLs')

    # get the relevant information from each advert
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'}
    data = []
    counter = 0
    print('Adverts parsed (time per batch of 200)...')
    with open(target_filepath, 'r') as file:
        for i, url in enumerate(file):
            counter += 1
            page = requests.get(url.strip(), headers=headers) 
            # check if ad still exists
            if page.status_code != 200:
                warnings.warn(f'\nSomething went wrong with url number {i+1}: {url} \tStatus code: {page.status_code}')
                print('continuing...')
                continue

            page_soup = BeautifulSoup(page.content, 'html.parser')
            # get a couple of specific things
            property_title_span = page_soup.find('span', class_='property-card-immotype-title')
            property_title_children = property_title_span.findChildren('span')
            type_of_property = property_title_children[0].text.strip()
            locality = property_title_children[-1].text.strip()

            # get everything in the characteristics block of the page
            try:
                _characteristics_container_div = page_soup.find('div', class_='characteristics-container')
                characteristics_dict = _scan_characteristics_block(_characteristics_container_div)
            except:
                print(f"URL number {i+1} might have no info.")
            else:
                characteristics_dict['Property Type'] = type_of_property
                characteristics_dict['Locality'] = locality

                data.append(characteristics_dict)

            # progress print, as usual
            if counter % 200 == 0:
                batch_et_time = time.time()
                print(counter, f"\t({round(batch_et_time - batch_st_time, 2)} s)")
                batch_st_time = batch_et_time

    file.close()

    # turn the whole thing into a dataframe to save it as a CSV for future reference
    df = pd.DataFrame(data)
    csv_path = os.path.dirname(os.path.abspath(__file__)) + '/raw_datasets/' + f'data_{target_timestamp}.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8')

    et_time = time.time()
    print(f"Successfully saved data to CSV file with path '{csv_path}'.")
    print(f"This process took {round(et_time - st_time, 2)} seconds.")

    return

def _scan_characteristics_block(container):

    blocks = container.find_all('div', class_='characteristics-block')
    data = {}
    # scan through each block of characteristics logging data in a dictionary
    for block in blocks:
        block_direct_children = block.findChildren('div', recursive=False)
        for child in block_direct_children[1:]:
            label = child.find('span', class_='characteristics-item-label').text.strip()
            value = child.find('span', class_='characteristics-item-value').text.strip()
            data[label] = value
    
    return data


################################################
### Test code
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
    # extract_athomelu_entries()
    # get_data()
    # _find_characteristics()
    # _test()
    # _setup_directory()
    # _gather_subset()
    pass