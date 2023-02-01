import pandas as pd
import os
from datetime import datetime
import numpy as np

"""
    1. Keep only columns with enough data to be worth keeping.
    2. transform columns into appropriate types and remove superfluous information.
    3. pre-process the data into ML-friendly format (one-hot, categorise using numbers, etc)
    4.
"""

################################################
### Functions for cleaning the data

def cleanup():

    # select the most up to date set of raw data
    current_filepath = os.path.abspath(os.path.abspath(__file__))
    csv_filepath = current_filepath + '/raw_datasets/'
    list_of_files = os.listdir(csv_filepath)
    # raise error if there are no files in the directory
    if not len(list_of_files):
        raise Exception(f'No files exist in {csv_filepath}')

    file_timestamps = [int(x.split('_')[-1][:-4]) for x in list_of_files]
    target = str(max(file_timestamps))
    target_filepath = csv_filepath + [file for file in list_of_files if target in file][0]

    # import raw data
    df = pd.read_csv(target_filepath)



    # merge 'indoor' data into 'closed' parking spaces column
    if 'Indoor parking space(s)' and 'Closed parking space' in df.columns:
        df['Closed parking space'] = df['Closed parking space'].fillna(df['Indoor parking space(s)'])
        df.drop('Indoor parking space(s)', axis=1, inplace=True)
    df['Closed parking space'] = df['Closed parking space'].apply(_nan_to_int)

    # remove any entries with no sale price
    df.drop(df[df['Sale price'].isnull()].index, inplace=True)
    # convert sale price into an integer value
    df['Sale price'] = df['Sale price'].apply(_str_to_int)
    # remove entries with sale price below 200k (don't want to include garages)
    cutoff = 200000
    df.drop(df[df['Sale price'] < cutoff].index, inplace=True)

    # convert Livable surface column into float
    df['Livable surface'] = df['Livable surface'].apply(lambda x: float(x.split(' ')[0].replace(',', '')))

    # convert Land into float value, replace NaNs with 0
    df['Land'] = df['Land'].apply(_str_to_float)

    # replace NaNs in num of bedrooms with median value (2), and convert to int
    num_bedrooms_median = df['Number of bedrooms'].median()
    df.loc[df['Number of bedrooms'].isnull(), 'Number of bedrooms'] = num_bedrooms_median
    df['Number of bedrooms'] = df['Number of bedrooms'].astype(int)

    # for year of construction:
    # fix entries from braindead people (e.g., year of construction: 12)
    current_year = datetime.today().year
    df.loc[df['Year of construction'] < 1000, 'Year of construction'] = current_year - df['Year of construction']
    # replace NaNs with median
    construction_year_median = df['Year of construction'].median()
    df.loc[df['Year of construction'].isnull(), 'Year of construction'] = construction_year_median
    # bucketise
    construction_year_bins = [0, 1870, 1920, 1950, 1970, 1980, 1990, 2000, 2010, 2015, 2020, current_year] 
    construction_year_labels = np.arange(1,len(construction_year_bins))
    df['Year of construction'] = pd.cut(df['Year of construction'], bins=construction_year_bins, labels=construction_year_labels)

    # bucketise renovation year also
    renovation_year_bins = [0, 1980, 1990, 2000, 2010, 2015, 2020, 2023]
    renovation_year_labels = np.arange(1,len(renovation_year_bins))
    df['Renovation year'] = pd.cut(df['Renovation year'], bins=renovation_year_bins, labels=renovation_year_labels)

    # convert Terrace into float
    df['Terrace'] = df['Terrace'].apply(_str_to_float)

    # convert Open parking space into int, turning NaNs into 0 and 'Yes' into 1
    df['Open parking space'] = df['Open parking space'].apply(_yesint_to_int)

    # convert Energy class NaNs into 'NC' and strip the numbers from those that have them
    df['Energy class'] = df['Energy class'].apply(_energy_class_convert)

    # same thing with Thermal insulation class
    df['Thermal insulation class'] = df['Thermal insulation class'].apply(_energy_class_convert)

    # convert Open kitchen into binary int
    df['Open kitchen'] = df['Open kitchen'].apply(_binary_int)

    # merge values from 'Bathooms' into the Bathroom column and convert NaNs into 0
    if 'Bathooms' and 'Bathroom' in df.columns:
        df['Bathroom'] = df['Bathroom'].fillna(df['Bathooms'])
        df.drop('Bathooms', axis=1, inplace=True)
    df['Bathroom'] = df['Bathroom'].apply(_nan_to_int)

    # convert Basement column into binary int
    df['Basement'] = df['Basement'].apply(_binary_int)

    # convert Property's floor NaNs into 0's and the rest into int
    df['Property\'s floor'] = df['Property\'s floor'].apply(_nan_to_int)

    # convert Lift column to binary int
    df['Lift'] = df['Lift'].apply(_binary_int)

    # convert Fitted kitchen column to binary int
    df['Fitted kitchen'] = df['Fitted kitchen'].apply(_binary_int)
    
    # convert Separate kitchen column to binary int
    df['Separate kitchen'] = df['Separate kitchen'].apply(_binary_int)

    # convert Restroom column into int
    df['Restroom'] = df['Restroom'].apply(_yesint_to_int)

    # convert Laundry column to binary int
    df['Laundry'] = df['Laundry'].apply(_binary_int)

    # combine 'Shower room' and 'Shower rooms' into one column, int
    if 'Shower rooms' and 'Shower room' in df.columns:
        df['Shower rooms'] = df['Shower rooms'].fillna(df['Shower room'])
        df.drop('Shower room', axis=1, inplace=True)
    df['Shower rooms'] = df['Shower rooms'].apply(_nan_to_int)

    # convert Balcony column into float
    df['Balcony'] = df['Balcony'].apply(_str_to_float)

    # convert Pets accepted column to binary int
    df['Pets accepted'] = df['Pets accepted'].apply(_binary_int)

    # convert Swimming pool column to binary int
    df['Swimming pool'] = df['Swimming pool'].apply(_binary_int)

    # convert Sauna column to binary int
    df['Sauna'] = df['Sauna'].apply(_binary_int)

    # convert Solar panels column to binary int
    df['Solar panels'] = df['Solar panels'].apply(_binary_int)

    # convert Garden column into float
    df['Garden'] = df['Garden'].apply(_str_to_float)

    # convert Fireplace column to binary int
    df['Fireplace'] = df['Fireplace'].apply(_binary_int)

    # convert Attic column to binary int
    df['Attic'] = df['Attic'].apply(_binary_int)



    # drop all useless columns (maybe do this at the end and selecting which ones to keep instead of which to remove)
    superfluous_columns = ['Number of rooms', 'Geothermal heating', 'Electric heating', 'Total floors', 'Acces for mobility-impared people', 
        'Acces for mobility-impared people', 'Monthly charges', 'Fuel heating', 'Pump heating', 'Convertible attic', 'Renovated', 'Availability',
        'Living room', 'Parquet', 'Gas heating']
    df.drop(superfluous_columns, axis=1, inplace=True)




    new_column_names = ['sale_price_m2', 'livable_surface_m2', 'land_ares', 'terrace_m2', 'balcony_m2']

    return


def _nan_to_int(x):
    if pd.isnull(x) or x > 50:
        return 0
    else:
        return int(x)

def _str_to_int(x):
    try:
        return int(str(x).split(' ')[0].replace(',', ''))
    except:
        return 0

def _str_to_float(x):
    if pd.isnull(x):
        return 0.0
    else:
        return float(x.split(' ')[0].replace(',', ''))

def _binary_int(x):
    if pd.isnull(x):
        return 0
    else:
        return 1

def _energy_class_convert(x):
    if pd.isnull(x) or x == 'NS':
        return 'NC'
    else:
        return ''.join(filter(str.isalpha, x))

def _yesint_to_int(x):
    if pd.isnull(x):
        return 0
    elif x == 'Yes':
        return 1
    else:
        return int(x)






if __name__ == '__main__':

    pass
