import pandas as pd
from pathlib import Path
import numpy as np
import sys

def format_suburb_names(s_name, lower= False, upper = False):
    """
    Formats suburb names from data provided by ABS

    Args:
        s_name: (str) ABS suburb name
        lower: (int) number of words to remove from the start of s_name
        upper: (int) number of words to remove from the end of s_name
    
    Returns:
        (str) Optionally shortened string, with instances of '(ACT)' removed
    """       
    if not lower and not upper:
        s_name = s_name.split()
    else:
        s_name = s_name.split()[lower:upper]
    s_name_filtered = [s for s in s_name if s != '(ACT)']
    return ' '.join(s_name_filtered)


def create_suburb_df(f):
    """
    Reads ABS suburb statistics files in suburb_path to df and concantenates to dataframe 

    Args:
        f: (str) File name in suburb_path
    
    Returns:
        suburb: (pd.Dataframe) Current dataframe including new data from f
    """    
    suburb = pd.read_csv(suburb_path / f)
    suburb['Suburb'] = format_suburb_names(f, lower= 2, upper = -2)
    return suburb


def combine_parkes_ch_suburbs(df, group_list):
    """
    Combines Parkes - North and Parkes - South ABS suburbs to form Parkes & Capital Hill

    Args:
        df: (pd.Dataframe) ABS sourced dataframe
        group_list: (list(str)) Specifies df columns to be used for grouping
    
    Returns:
        df: (pd.Dataframe) Updated dataframe
    """  
    comb_df = df[(df['Suburb'] == 'Parkes - North') | (df['Suburb'] == 'Parkes - South')]
    comb_df = comb_df.groupby(group_list).sum().reset_index()
    parks_ch = comb_df[comb_df['Suburb'] == 'Parkes - NorthParkes - South'].copy()
    parks_ch['Suburb'] = 'Parkes & Capital Hill'
    df = df[(df['Suburb'] != 'Parkes - North') & (df['Suburb'] != 'Parkes - South')]
    df = pd.concat([df, parks_ch], ignore_index=True)
    return df


def group_crimes(df, group_list):
    """
    Groups crime statistics by crime similarity

    Args:
        df: (pd.Dataframe) Crime dataframe with original crime names
        group_list: (list(str)) Specifies df columns to be used for grouping
    
    Returns:
        df: (pd.Dataframe) Grouped crime dataframe
    """

    burglary = df[(df['Crime'] == 'Burglary - Dwellings') | (df['Crime'] == 'Burglary - Other') | 
                        (df['Crime'] == 'Burglary - Shops')]
    burglary = burglary.groupby(group_list).sum().reset_index()
    burglary['Crime'] = 'Burglary'
    df = pd.concat([df, burglary], ignore_index=True)

    robbery = df[(df['Crime'] == 'Robbery - Armed') | (df['Crime'] == 'Robbery - Other')]
    robbery = robbery.groupby(group_list).sum().reset_index()
    robbery['Crime'] = 'Robbery'
    df = pd.concat([df, robbery], ignore_index=True)

    theft = df[(df['Crime'] == 'Theft - Motor Vehicles') | (df['Crime'] == 'Theft - Other')]
    theft = theft.groupby(group_list).sum().reset_index()
    theft['Crime'] = 'Theft'
    df = pd.concat([df, theft], ignore_index=True)

    other = df[(df['Crime'] == 'Other - Against A Person') | (df['Crime'] == 'Other')]
    other = other.groupby(group_list).sum().reset_index()
    other['Crime'] = 'Other Crime'
    df = pd.concat([df, other], ignore_index=True)

    tins = df[(df['Crime'] == 'TINs - Speeding') | (df['Crime'] == 'TINs - Mobile Use') | 
                    (df['Crime'] == 'TINs - Seatbelts') | (df['Crime'] == 'TINs - Other')]
    tins = tins.groupby(group_list).sum().reset_index()
    tins['Crime'] = 'TINs'
    df = pd.concat([df, tins], ignore_index=True)

    # remove crimes which form a group
    df = df[df['Crime'].isin(['Burglary', 'Robbery', 'Theft', 'Other', 
                              'TINs', 'Homicide', 'Family Violence', 'Assault', 
                              'Sexual Assault', 'Property Damage', 'CINs'])]
    return df


def add_regions(df, region_df):
    """
    Creates 'Region' column corresponding to 'Suburb' column

    Args:
        df: (pd.Dataframe) Dataframe containing 'Suburb' column
        region_df: (pd.Dataframe) Dataframe listing suburbs (SA2) and their regions (SA3)
    
    Returns:
        merged_df: (pd.Dataframe) df with added 'Region' column
    """
    merged_df = pd.merge(df, region_df, how='left', on='Suburb')  
    return merged_df


def save_region_suburb(df, group_cols, title, is_crime = False):
    """
    Reorders and saves original dataframe, alongside the same dataframe grouped by 'Region'
    Crime data not saved to final processed path to allow for further manipulation
    Args:
        df: (pd.Dataframe) Dataframe containing 'Suburb' and 'Region' columns
        group_list: (list(str)) Specifies df columns to be used for grouping
        title: (str) Save file name prefix
        is_crime: (binary) Determines file save path
    """
    if is_crime:
        save_paths = [processed_path, processed_path]
    else:
        save_paths = [final_path_region, final_path_suburb]

    region_df = df.groupby(group_cols).sum().reset_index().drop(columns=['Suburb'])
    name = title + '_region.csv' 
    region_df.to_csv(save_paths[0] / name, index=False)

    col_order = region_df.columns.tolist()
    col_order.insert(1, 'Suburb')
    df = df[col_order]
    name = title + '_suburb.csv' 
    df.to_csv(save_paths[1] / name, index=False)


### PROCESS SUBURB DATA ###
processed_path = Path('data/processed')
final_path_region = Path('data/processed/final/region')
final_path_suburb = Path('data/processed/final/suburb')
final_path_suburb.mkdir(parents=True, exist_ok=True)
final_path_region.mkdir(parents=True, exist_ok=True)

raw_path = Path('data/raw')
suburb_path = Path('data/raw/suburb')

# combine seperate suburb files
raw_f_names = [f.name for f in suburb_path.iterdir() if f.is_file()]
sa2_f_names = [f_name for f_name in raw_f_names if 'Region summary_' in f_name]

suburb_df = create_suburb_df(sa2_f_names[0])
for s in sa2_f_names[1:]:
    suburb_df = pd.concat([suburb_df, create_suburb_df(s)], ignore_index=True)

# combine Parkes and Capitol Hill areas
# warning: incomplete socioeconomic data for Parkes 
suburb_df = combine_parkes_ch_suburbs(suburb_df, ['Measure Code', 'Parent Description', 'Description'])


### PROCESS POPULATION DATA ###

pop = pd.read_excel(raw_path / 'population_estimates_aus.xlsx', sheet_name=None)
pop_df = pop['Table 1']
pop_df.columns = pop_df.iloc[3]
pop_df = pop_df.iloc[5:, [1] + list(range(9, len(pop_df.columns)))]
pop_df.columns = ['State', 'Suburb'] + list(pop_df.columns[2:])
pop_df = pop_df[pop_df['State'] == 'Australian Capital Territory']

pop_df['Suburb'] = pop_df['Suburb'].apply(format_suburb_names)
pop_df = combine_parkes_ch_suburbs(pop_df, ['State'])
pop_df = pop_df.drop(columns='State')
# remove suburbs with no/unknown population from start of date range
pop_df = pop_df[pop_df[2014] != 0]

### PROCESS CRIME DATA ###

crime_data = pd.read_excel((raw_path / 'Website_Qtrly_Jun25.xlsx'), sheet_name=None)
processed_crime = {}
# format suburb crime data
for k in crime_data.keys():
    suburb_data = crime_data[k].iloc[2:]

    #set temporal column names
    col_names = suburb_data.iloc[1].copy().tolist()
    col_names[0] = 'Suburb'
    suburb_data.columns = col_names

    # remove temporal rows
    suburb_data = suburb_data[suburb_data.iloc[:,0].notna()].reset_index(drop=True)
    # set crime type as column
    suburb_data['Crime'] = np.where(suburb_data.iloc[:,1:].isna().all(axis=1), suburb_data.iloc[:,0], np.nan)
    suburb_data['Crime'] = suburb_data['Crime'].ffill()
    suburb_data = suburb_data[suburb_data.iloc[:,1:].notna().all(axis=1)]
    suburb_data = suburb_data[suburb_data['Suburb'] != 'Total']
    suburb_data['Region'] = k
    processed_crime[k] = suburb_data

# combine data into one df
crime_df = pd.concat(processed_crime.values(), ignore_index=True)
crime_df.columns = np.where((crime_df.columns != 'Suburb') | (crime_df.columns != 'Crime'), 
                            crime_df.columns.str.replace(r'\s+\S+$', '', regex=True), crime_df.columns)

# format crime names
crime_df['Crime'] = crime_df['Crime'].str.title()

crime_df['Crime'] = crime_df['Crime'].replace({'Assault - Fv': 'Family Violence', 'Assault - Non-Fv': 'Assault', 
                                                'Other Offences Against A Person': 'Other - Against A Person',
                                                'Burglary Dwellings': 'Burglary - Dwellings',
                                                'Burglary Shops': 'Burglary - Shops', 
                                                'Burglary Other': 'Burglary - Other',
                                                'Motor Vehicle Theft' : 'Theft - Motor Vehicles',
                                                'Theft (Excluding Motor Vehicles)': 'Theft - Other',
                                                'Other Offences': 'Other', 'Tins Speeding': 'TINs - Speeding',
                                                'Tins Mobile Use': 'TINs - Mobile Use', 'Tins Seatbelts': 'TINs - Seatbelts',
                                                'Tins Other': 'TINs - Other', 'Cins': 'CINs'})

# format crime suburb names
crime_df['Suburb'] = crime_df['Suburb'].str.title()
crime_df['Suburb'] = np.where(crime_df['Suburb'] == 'Mckellar', 'McKellar', crime_df['Suburb'])
crime_df['Suburb'] = np.where(crime_df['Suburb'] == 'City', 'Civic', crime_df['Suburb'])
# combine Parkes and Capitol Hill areas
crime_pch = crime_df[(crime_df['Suburb'] == 'Parkes') | (crime_df['Suburb'] == 'Capital Hill')]
crime_pch = crime_pch.groupby('Crime').sum().reset_index()
# warning: crossover between inner north and south 
crime_pch['Suburb'] = 'Parkes & Capital Hill'
crime_pch['Region'] = 'Inner South'
crime_df = pd.concat([crime_df, crime_pch], ignore_index=True)
crime_df = crime_df[(crime_df['Suburb'] != 'Parkes') & (crime_df['Suburb'] != 'Capital Hill')]
# remove Hume 
crime_df = crime_df[crime_df['Region'] != 'Other']

### PROCESS ALL ###

# find suburbs with population counts available
all_suburbs = suburb_df['Suburb'].unique()
pop_suburbs = pop_df['Suburb'].unique()

# filter suburb statistics based on data availability
suburb_df['Total Description'] = suburb_df['Parent Description'] + ' : ' + suburb_df['Description']
suburb_df = suburb_df.drop(columns=['Measure Code', 'Parent Description', 'Description', '2015', 
                                    '2017', '2018', '2019', '2020','2022', '2023', '2024','2025'])
suburb_df = suburb_df.dropna()
desc_counts = suburb_df['Total Description'].value_counts()
keep_stats = desc_counts[desc_counts == desc_counts.max()].index
suburb_df = suburb_df[suburb_df['Total Description'].isin(keep_stats)]

# keep only data for suburbs found in intersection of suburb and crime dataframes
remove_s_suburb = list(set(suburb_df['Suburb'].unique()) - set(pop_df['Suburb'].unique()))
suburb_df = suburb_df[~suburb_df['Suburb'].isin(remove_s_suburb)]
remove_s_crime = list(set(crime_df['Suburb'].unique()) - set(suburb_df['Suburb'].unique()))
crime_df = crime_df[~crime_df['Suburb'].isin(remove_s_crime)]
remove_s_pop = list(set(pop_df['Suburb'].unique()) - set(crime_df['Suburb'].unique()))
pop_df = pop_df[~pop_df['Suburb'].isin(remove_s_pop)]
remove_s_suburb = list(set(suburb_df['Suburb'].unique()) - set(pop_df['Suburb'].unique()))
suburb_df = suburb_df[~suburb_df['Suburb'].isin(remove_s_suburb)]

# reshape into long format
crime_long = crime_df.melt( id_vars=['Suburb', 'Crime', 'Region'], var_name='Year_Quarter', value_name='Number')
crime_long[['Year', 'Quarter']] = crime_long['Year_Quarter'].str.split(' ', expand=True)
crime_long = crime_long.drop(columns=['Year_Quarter'])
crime_long['Year'] = crime_long['Year'].astype(int)
pop_long = pop_df.melt( id_vars='Suburb', var_name='Year', value_name='Population')
pop_long = pop_long[pop_long['Year'].isin(crime_long['Year'])]
suburb_long = suburb_df.melt( id_vars=['Suburb', 'Total Description'], var_name='Year', value_name='Value')

# add region data where missing
region_suburbs = crime_long[['Region', 'Suburb']].drop_duplicates(['Region', 'Suburb'])
pop_long = add_regions(pop_long, region_suburbs)
suburb_long = add_regions(suburb_long, region_suburbs)


### EXPORT DATA ###

crime_long_grouped = group_crimes(crime_long.copy(), ['Region', 'Suburb', 'Year', 'Quarter'])
save_region_suburb(crime_long, ['Region', 'Crime', 'Year', 'Quarter'], 'crime', True)
save_region_suburb(crime_long_grouped, ['Region', 'Crime', 'Year', 'Quarter'], 'crime_grouped', True)
save_region_suburb(pop_long, ['Region', 'Year'], 'population')
save_region_suburb(suburb_long, ['Region', 'Year', 'Total Description'], 'suburb_stats')
