import pandas as pd
import os
from datetime import datetime
import numpy as np

from utils import _setup_directory, _find_file


################################################
### Functions for cleaning the data
def cleanup() -> None:
    """Oh"""

    # quick setup
    _setup_directory()

    # select the most up to date set of raw data
    target_filepath, target_timestamp = _find_file('raw_datasets')

    # import raw data
    df = pd.read_csv(target_filepath)



    # remove any entries with no sale price
    df.drop(df[df['Sale price'].isnull()].index, inplace=True)
    # convert sale price into an integer value
    df['Sale price'] = df['Sale price'].apply(_str_to_float)
    # remove entries with sale price below 200k (don't want to include garages)
    cutoff = 200000
    df.drop(df[df['Sale price'] < cutoff].index, inplace=True)

    # convert Livable surface column into float
    df['Livable surface'] = df['Livable surface'].apply(_str_to_float)

    # convert Land into float value, replace NaNs with 0
    df['Land'] = df['Land'].apply(_str_to_float)

    # replace NaNs in num of bedrooms with median value (2), and convert to int
    num_bedrooms_median = df['Number of bedrooms'].median()
    df.loc[df['Number of bedrooms'].isnull(), 'Number of bedrooms'] = num_bedrooms_median

    # for year of construction:
    # fix entries from braindead people (e.g., year of construction: 12)
    current_year = datetime.today().year
    df.loc[df['Year of construction'] < 1000, 'Year of construction'] = current_year - df['Year of construction']
    # replace NaNs with median
    construction_year_median = df['Year of construction'].median()
    df.loc[df['Year of construction'].isnull(), 'Year of construction'] = construction_year_median

    # convert Terrace into float
    df['Terrace'] = df['Terrace'].apply(_str_to_float)

    # merge 'indoor' data into 'closed' parking spaces column
    if 'Indoor parking space(s)' and 'Closed parking space' in df.columns:
        df['Closed parking space'] = df['Closed parking space'].fillna(df['Indoor parking space(s)'])
        df.drop('Indoor parking space(s)', axis=1, inplace=True)
    df['Closed parking space'] = df['Closed parking space'].apply(_nan_to_float)

    # convert Open parking space into int, turning NaNs into 0 and 'Yes' into 1
    df['Open parking space'] = df['Open parking space'].apply(_yesint_to_float)

    # convert Energy class NaNs into 'NC' and strip the numbers from those that have them
    df['Energy class'] = df['Energy class'].apply(_energy_class_convert)

    # same thing with Thermal insulation class
    df['Thermal insulation class'] = df['Thermal insulation class'].apply(_energy_class_convert)

    # convert Open kitchen into binary int
    df['Open kitchen'] = df['Open kitchen'].apply(_binary_cat)

    # merge values from 'Bathooms' into the Bathroom column and convert NaNs into 0
    if 'Bathooms' and 'Bathroom' in df.columns:
        df['Bathroom'] = df['Bathroom'].fillna(df['Bathooms'])
        df.drop('Bathooms', axis=1, inplace=True)
    df['Bathroom'] = df['Bathroom'].apply(_nan_to_float)

    # convert Basement column into binary int
    df['Basement'] = df['Basement'].apply(_binary_cat)

    # convert Property's floor NaNs into 0's and the rest into int
    df['Property\'s floor'] = df['Property\'s floor'].apply(_nan_to_float)

    # convert Lift column to binary int
    df['Lift'] = df['Lift'].apply(_binary_cat)

    # convert Fitted kitchen column to binary int
    df['Fitted kitchen'] = df['Fitted kitchen'].apply(_binary_cat)
    
    # convert Separate kitchen column to binary int
    df['Separate kitchen'] = df['Separate kitchen'].apply(_binary_cat)

    # convert Restroom column into int
    df['Restroom'] = df['Restroom'].apply(_yesint_to_float)

    # convert Laundry column to binary int
    df['Laundry'] = df['Laundry'].apply(_binary_cat)

    # combine 'Shower room' and 'Shower rooms' into one column, int
    if 'Shower rooms' and 'Shower room' in df.columns:
        df['Shower rooms'] = df['Shower rooms'].fillna(df['Shower room'])
        df.drop('Shower room', axis=1, inplace=True)
    df['Shower rooms'] = df['Shower rooms'].apply(_nan_to_float)

    # convert Balcony column into float
    df['Balcony'] = df['Balcony'].apply(_str_to_float)

    # convert Pets accepted column to binary int
    df['Pets accepted'] = df['Pets accepted'].apply(_binary_cat)

    # convert Swimming pool column to binary int
    df['Swimming pool'] = df['Swimming pool'].apply(_binary_cat)

    # convert Sauna column to binary int
    df['Sauna'] = df['Sauna'].apply(_binary_cat)

    # convert Solar panels column to binary int
    df['Solar panels'] = df['Solar panels'].apply(_binary_cat)

    # convert Garden column into float
    df['Garden'] = df['Garden'].apply(_str_to_float)

    # convert Fireplace column to binary int
    df['Fireplace'] = df['Fireplace'].apply(_binary_cat)

    # convert Attic column to binary int
    df['Attic'] = df['Attic'].apply(_binary_cat)

    
    # drop all other columns
    columns_keep = ['Sale price', 'Property Type', 'Livable surface', 'Land', 'Number of bedrooms', 'Year of construction', 'Renovation year', 
        'Terrace', 'Closed parking space', 'Open parking space', 'Energy class', 'Thermal insulation class', 'Open kitchen', 
        'Bathroom', 'Basement', 'Property\'s floor', 'Lift', 'Fitted kitchen', 'Separate kitchen', 'Restroom', 'Laundry', 
        'Shower rooms', 'Balcony', 'Pets accepted', 'Swimming pool', 'Sauna', 'Solar panels', 'Garden', 'Fireplace', 'Attic']
    for col in df.columns:
        if col not in columns_keep:
            df.drop(col, axis=1, inplace=True)

    # save the resulting dataset
    csv_path = os.path.dirname(os.path.abspath(__file__)) + '/clean_datasets/' + f'clean_data_{target_timestamp}.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8')

    return None


### Support functions
def _nan_to_float(x):
    if pd.isnull(x) or x > 50:
        return 0.
    else:
        return float(x)

def _str_to_float(x):
    if pd.isnull(x):
        return 0.
    else:
        try:
            return float(x.split(' ')[0].replace(',', ''))
        except:
            return 0.

def _binary_cat(x):
    if pd.isnull(x):
        return 0.
    else:
        return 1.

def _energy_class_convert(x):
    if pd.isnull(x) or x == 'NS':
        return 'NC'
    else:
        return ''.join(filter(str.isalpha, x))

def _yesint_to_float(x):
    if pd.isnull(x):
        return 0.
    elif x == 'Yes':
        return 1.
    else:
        return float(x)




if __name__ == '__main__':
    cleanup()
    pass
