import pandas as pd
pd.set_option("display.max.columns", None)
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning) # remove some useless pandas warnings
import numpy as np
import datetime
import matplotlib.pyplot as plt

# for encoding categorical variables
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# helper to find most recent files
from utils import _find_file

## constants
SALE_PRICE_CUTOFF = 140000
NAN_FRACTION_CUTOFF = 0.7
# some locations outside lux that I spotted manually that don't belong here
INVALID_LOCALITIES = [
    'Marbella',
    'Daya vieja',
    'Las chapas de marbella',
    'Brescia', 
    'Pombal',
    'Siniscola',
    'Nueva andalucia', 
    'Cala pi'
]
# most common localities determined during exploration, use this list to reduce cardinality of 'locality' feature
# other localities had fewer than 50 records in the entire set
MAIN_LOCALITIES = [
    'Belair', 'Esch-sur-Alzette', 'Mamer', 'Dudelange', 'Erpeldange',
    'Mersch', 'Centre ville', 'Bascharage', 'Ettelbruck', 'Bertrange',
    'Bonnevoie', 'Differdange', 'Kirchberg', 'Schifflange', 'Junglinster',
    'Strassen', 'Niederkorn', 'Hautcharage', 'Limpertsberg', 'Merl',
    'Belvaux', 'Gasperich', 'Belval', 'Rodange', 'Rumelange',
    'Mondorf-Les-Bains', 'Steinfort', 'Wiltz', 'Bissen', 'Bridel',
    'Diekirch', 'Pétange', 'Remich', 'Howald', 'Kayl', 'Kehlen', 'Steinsel',
    'Walferdange', 'Hesperange', 'Moutfort', 'Gare', 'Hollerich',
    'Capellen', 'Bettembourg', 'Cents', 'Eich', 'Bereldange',
    'Lorentzweiler', 'Heisdorf', 'Stegen', 'Contern'
]
# features determined too sparse during exploration
OVERLY_SPARSE_FEATURES = [
    'number_of_rooms', 'heat_pump', 'open_kitchen', 'gas_heating',
    'fitted_kitchen', 'balcony', 'mandate_type', 'total_floors',
    'renovated', 'year_of_renovation', 'orientation', 'floor_heating',
    'optical_fiber', 'fireplace', 'attic', 'monthly_charges',
    'bike_storage', 'telephone_line', 'oil-fired_heating', 'solar_panels',
    'electricity_plug_in_the_parking', 'indoor_parking_space',
    'pellets_heating', 'coaxial_cable', 'swimming_pool', 'pets_allowed',
    'reduced_mobility_access', 'ethernet_network', 'photovoltaic',
    'electric_heating', 'alarm_system', 'emphyteutic_lease',
    'air_conditioning', 'wine_cellar', 'safe_box', 'geothermal_heating',
    'fire_alarm_network', 'video_monitoring', 'archives', 'partitioning',
    'locker_rooms', 'affordable_housing', 'access_control_systems',
    'computer_room', 'agency_commission', 'technical_floor',
    'freight_elevator', 'false_floor', 'life_annuity_sale', 'floor_ducts',
    'perimeter_ducts', 'canteen', 'cleaning_service', 'cable_tv'
]
# features determined useless during exploration
USELESS_FEATURES = [
    'availability', # either meaningless or already effectively encoded into year_of_construction 
]
# for convenience
FEATURES_TO_REMOVE = OVERLY_SPARSE_FEATURES + USELESS_FEATURES


def label_based_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    0th step of removing invalid data and reformatting labels before splitting labels from features.
    Removes records with Null labels as well as those with invalid localities.
    Makes all column names lowercase and swaps whitespaces with underscores.
    Converts string label values into numerical values (float).

    Parameters
    ----------
    df: pd.DataFrame
        DF containing entire raw dataset.

    Returns
    -------
    pd.DataFrame
        Same DF with potentially fewer records and reformatted column names and label data.
    """

    og_shape = df.shape
    print(f"Input data shape: {og_shape}")
    print("Cleaning data by removing invalid records.")

    #-----# 1. Quality of life changes #-----#
    
    # make columns lowercase and replace spaces with underscores
    # also remove the "(s)" at the end of one of them, it's annoying
    df.columns = (df.columns
                  .str.replace(' ', '_')
                  .str.lower()
                  .str.replace('(s)', ''))

    #-----# 2. Invalid label filtering and label formatting #-----#
    
    # convert sale_price to numerical value
    df['sale_price'] = df['sale_price'].str.replace(r"[€,]", '', regex=True).astype(float)

    # remove records with Null sale_price or with sale_price < SALE_PRICE_CUTOFF (140k)
    sale_price_mask = (df['sale_price'].isna() == False) & (df['sale_price'] >= SALE_PRICE_CUTOFF)
    n_removed = og_shape[0] - sale_price_mask.sum()
    print(f"Removing records where label ('sale_price') is Null or lower than {SALE_PRICE_CUTOFF}:    {n_removed} records removed.")
    df = df[sale_price_mask]

    #-----# 3. Invalid record filtering #-----#

    # remove invalid records which pertain to properties outside of luxembourg
    invalid_locality_mask = df['locality'].isin(INVALID_LOCALITIES)
    bad_rows_df = df[invalid_locality_mask]
    print(f"Removing records from locations outside of Luxembourg:    {invalid_locality_mask.sum()} records removed.")
    df = df.drop(bad_rows_df.index) 

    print(f"Cleaned data shape: {df.shape}")
    return df #.reset_index(drop=True) ###############################

def format_feature_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data cleaning process. Takes care of formatting and harmonising the raw feature data collected by the scraper.
    Remaps Null values as appropriate depending on the specific type of data contained in the feature,
    and creates {feature}_missingflag columns to indicate records that had missing data which was filled in
    in some way. 
    Introduces Nulls into certain numerical features for easy imputation in a later stage. 

    Parameters
    ----------
    df: pd.DataFrame
        DF containing raw feature data as extracted by the scraper module.
    
    Returns
    -------
    pd.DataFrame
        Same DF with reformatted columns, and additional 'missing flag' columns.

    """
    
    og_shape = df.shape
    print(f"Formatting feature data. Input shape: {og_shape}")

    #-----# 1. Generate new "missing data" indicator features #-----#

    # create flag columns for those that contain nulls
    # {feature}_missingflag columns will indicate whether the original {feature} value was missing for a given record record
    print("Generating new '{feature}_missingflag' columns to flag missing data in a given record:")
    for colname, colseries in df.items():
        if colseries.isna().sum():
            df[f"{colname}_missingflag"] = colseries.isna().astype(int)
    print(f"\t {df.shape[1] - og_shape[1]} new columns generated.")

    #-----# 2. Categorical Features: Reformat/clean/harmonise values #-----#

    # remove "Luxembourg-" prefix from localities
    df.locality = df.locality.str.replace("Luxembourg-", "")
    # make location in brackets main location: some locations specified as "small town (commune)"
    # split according to '(', take last string except last character ')'
    main_locality_replace = lambda x: x.split('(')[-1][:-1] if '(' in x else x
    df.locality = df.locality.apply(main_locality_replace)
    # finally, reduce cardinality of 'locality' feature to most common locations
    df.locality = df.locality.apply(lambda x: x if x in MAIN_LOCALITIES else 'other')

    # keep only the letter grading on the energy and thermal insulation classes (some specified as "196.1E", keep only "E")
    # if the string contains more than 1 character after keeping only letters it means it's "blank" or some other word to be replaced with NaN
    classcols = [col for col in df.columns if ('class' in col) and ('_missingflag' not in col)]
    for colname in classcols:
        df[colname] = df[colname].str.replace('[^a-zA-Z]', '', regex=True)
        df[colname] = df[colname].apply(lambda x: np.nan if len(str(x)) > 1 else x)
        # add newly assigned NaNs to corresponding _missingflag column as missing
        df.loc[df[colname].isna(), f"{colname}_missingflag"] = 1
        # assign 'Z' to missing values so they'll be ordered last in the categories
        df[colname] = df[colname].fillna('Z')

    # yes/no columns (yes/NaN actually but whatever): reformat into binary flag columns (1=yes, 0=no)
    yesnocols = [col for col in df.columns if (df[col] == 'Yes').sum()]
    for colname in yesnocols:
        # sometimes the 'garden' or 'terrace' features are filled with 'yes' instead of a surface value
        if any(df[colname].str.find('m²') > -1):
            yesnocols.remove(colname)
        else:
            df[colname] = (df[colname].str.lower() == 'yes').astype(float)

    
    #-----# 3. Numerical Features: dtype conversion/formatting #-----#

    # columns denoting areas (m^2, ares) need to be translated from string into numerical value
    # find all names of columns that contain 'm²' or 'ares'
    m2_cols = []
    ares_cols = []
    for colname, colseries in df.items():
        if not pd.api.types.is_numeric_dtype(colseries):
            if any(colseries.str.find('m²') > -1):
                m2_cols.append(colname)
            elif any(colseries.str.find('ares') > -1):
                ares_cols.append(colname)

    # additional complication: garden and terrace columns sometimes filled with "yes" instead of surface value
    for col in m2_cols:
        # First fill nulls with "0" to prevent later imputation
        df[col] = df[col].fillna('0')
        # replace Yes with NaN so it gets filled in later with the median
        df[col] = df[col].str.lower().replace('yes', np.nan)
        # remove units and turn into float
        df[col] = df[col].str.replace(r' m²|m|,', '', regex=True).astype(float)

    # simpler with ares
    for col in ares_cols:
        # First fill nulls with "0" to prevent later imputation
        df[col] = df[col].fillna('0')
        # remove units and commas and turn into float
        df[col] = df[col].str.replace(r' ares|,', '', regex=True).astype(float)

    # for garage and propert'y_floor, fill Nulls with 0 also, seems logical that an empty value means it is not applicable
    for col in ["garage", "property's_floor"]:
        df[col] = df[col].fillna(0)

    # add 2000 to the moron who put his construction year as just "12"
    df['year_of_construction'] = df['year_of_construction'].apply(lambda x: x+2000 if x < 1000 else x)
    # create new column for age_since_construction which is a more meaningful way of expressing it, then drop original
    print("Converting column 'year_of_construction' into 'age_since_construction'.")
    df['age_since_construction'] = datetime.datetime.today().year - df['year_of_construction']
    df = df.drop('year_of_construction', axis=1)
    
    print(f"Formatted feature data shape: {df.shape}")
    return df


def encode_categoricals(df: pd.DataFrame, encoders: dict[str, object]) -> pd.DataFrame:
    """
    Encodes categorical variables using 2 different strategies, both of which must be provided
    in the input 'encoders' dictionary.
    1. Ordinal Encoding: for features like energy class in which the order is meaningful.
    2. One-hot Encoding: for all other categorical features.

    Parameters
    ----------
    df: pd.DataFrame
        DF containing feature data where categorical features have string values (no nulls).
    encoders: dict[str, object]
        Dictionary containing the encoders pre-fitted to the training set, ready to perform
        encoder.transform(X) operations. 

    Returns
    -------
    pd.DataFrame
        Same DF where categorical features have been replaced with their encoded versions. 
    
    """

    # unpack encoder dict
    ordinal_encoder = encoders['ordinal_encoder']
    ordinal_columns = list(ordinal_encoder.feature_names_in_)
    onehot_encoder = encoders['onehot_encoder']
    onehot_columns = list(onehot_encoder.feature_names_in_)

    # apply transformations
    df[ordinal_columns] = ordinal_encoder.transform(df[ordinal_columns])

    onehot_array = onehot_encoder.transform(df[onehot_columns])
    onehot_df = pd.DataFrame(onehot_array, columns=onehot_encoder.get_feature_names_out(onehot_columns), index=df.index)
    df = pd.concat([df.drop(onehot_columns, axis=1), onehot_df], axis=1)

    return df

def impute_numericals(df: pd.DataFrame, impute_map: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Fills Null values in input DataFrame with group-specific values for all numerical features.
    E.g., replaces Nulls in, say, the 'terrace' column with different values depending on whether
    the property type is 'House' or 'Apartment'.

    Parameters
    ----------
    df: pd.DataFrame
        DF containing feature data where numerical features may contain Null values.
    impute_map: dict[str, pd.Series]
        Dictionary where the Keys are numerical feature names and the values are pd.Series 
        which map the different property_types (as indices) to the imputation value for each
        type.

    Returns
    -------
    pd.DataFrame
    """

    grouped = df.groupby('property_type', group_keys=False) # group_keys=False is important for the .apply to work correctly
    # loop through columns
    for colname in df.columns:
        # filter only numerical non-flag columns
        if pd.api.types.is_numeric_dtype(df[colname]) and (df[colname].nunique(dropna=False) > 2):
            # fill Nulls in each feature with group-specific values (e.g., median of the group)
            df[colname] = grouped[colname].apply(lambda group: group.fillna(impute_map[colname][group.name]))

    return df
