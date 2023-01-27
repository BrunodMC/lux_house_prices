import requests
import urllib.request
import json
from bs4 import BeautifulSoup
from datetime import datetime
import time
import os


# step 1: scrape website for properties for sale
# step 2: convert links to properties advertised into database entries with the relevant info

#####################################
### Functions to find all relevant articles
def extract_athomelu_entries():
    """Scrapes athome.lu, collecting the URL to every single property advertised in Luxembourg and writing them to a file.
        (took 21 minutes to run in my test)"""
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
    et_time = time.time()
    print(f"Successfully wrote all relevant URLs to file with path '{filepath}'.")
    print(f"This process took {et_time - st_time} seconds.")
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


#####################################
### Functions to extract the actual data from the relevant articles

""" TO DO:
        1. from a property's info page, extract useful info.
        2. 
"""
def get_data():

    # find the most up to date set of URLs
    current_filepath = os.path.dirname(os.path.abspath(__file__))
    URLs_filepath = current_filepath + '/extracted_URLs/'
    list_of_files = os.listdir(URLs_filepath)
    file_timestamps = [int(x.split('_')[-1][:-4]) for x in list_of_files]
    target = str(max(file_timestamps))
    target_filepath = URLs_filepath + [file for file in list_of_files if target in file][0]
    
    # get the relevant information from each advert
    with open(target_filepath, 'r') as file:
        for url in file:
            page = requests.get(url.strip()) ################################## put url back
            page_soup = BeautifulSoup(page.content, 'html.parser')
            print(page_soup.find_all('body')[0].encode('utf-8'))

            # page = urllib.request.urlopen(url) # .request('get', url) # urllib3.connection_from_url(url). .get(url)
            # jpage = json.load(page)
            # print(jpage)
            # page_soup = BeautifulSoup(page, 'lxml')
            # test = page_soup.find_all('div', class_='block-localisation-address')
            # print(page_soup.encode('utf-8'))
            # print(test)
            
            # get basic info (location, type of dwelling, price) 
            # address = page_soup.find_all("i")
            # print(address)
            break
    file.close()
    return





if __name__ == '__main__':
    get_data()
    