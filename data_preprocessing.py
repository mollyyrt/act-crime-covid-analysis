import pandas as pd
from pathlib import Path
import numpy as np
import sys
from src.preprocessing_funcs import (
    format_suburb_names,
    create_suburb_df,
    combine_parkes_ch_suburbs,
    group_crimes,
    add_regions, 
    save_region_suburb
)


raw_path = Path('data/raw')
suburb_path = Path('data/raw/suburb')
processed_path = Path('data/processed')

### PROCESS SUBURB DATA ###
# combine seperate suburb files
raw_f_names = [f.name for f in suburb_path.iterdir() if f.is_file()]
sa2_f_names = [f_name for f_name in raw_f_names if 'Region summary_' in f_name]

suburb_df = create_suburb_df(sa2_f_names[0], suburb_path)
for s in sa2_f_names[1:]:
    suburb_df = pd.concat([suburb_df, create_suburb_df(s, suburb_path)], ignore_index=True)

# combine Parkes and Capitol Hill areas
# warning: incomplete socioeconomic data for Parkes 
suburb_df = combine_parkes_ch_suburbs(suburb_df, ['Measure Code', 'Parent Description', 'Description'])


# filter suburb statistics based on data availability
suburb_df['Total Description'] = suburb_df['Parent Description'] + ' : ' + suburb_df['Description']
suburb_df = suburb_df.drop(columns=['Measure Code', 'Parent Description', 'Description', '2015', 
                                    '2017', '2018', '2019', '2020','2022', '2023', '2024','2025'])
suburb_df = suburb_df.dropna()
desc_counts = suburb_df['Total Description'].value_counts()
keep_stats = desc_counts[desc_counts == desc_counts.max()].index
suburb_df = suburb_df[suburb_df['Total Description'].isin(keep_stats)]
suburb_df = suburb_df[suburb_df['Total Description'].str.contains('(no.)', regex=False)]


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
suburb_long = suburb_df.melt( id_vars=['Suburb', 'Total Description'], var_name='Year', value_name='Number')
suburb_long['Year'] = suburb_long['Year'].astype(int)

# add region data where missing
region_suburbs = crime_long[['Region', 'Suburb']].drop_duplicates(['Region', 'Suburb'])
pop_long = add_regions(pop_long, region_suburbs)
suburb_long = add_regions(suburb_long, region_suburbs)

crime_long_grouped = group_crimes(crime_long.copy(), ['Region', 'Suburb', 'Year', 'Quarter'])
crime_suburb = pd.merge(crime_long_grouped, pop_long, on=['Region', 'Suburb', 'Year'], how='left').sort_values(by=['Region', 'Suburb', 'Crime', 'Year', 'Quarter'])
stats_suburb = pd.merge(suburb_long, pop_long, on=['Region', 'Suburb', 'Year'], how='inner').sort_values(by=['Region', 'Suburb', 'Total Description',])


### EXPORT DATA ###

save_region_suburb(crime_suburb, ['Crime', 'Year', 'Quarter'], 'crime', processed_path)
save_region_suburb(stats_suburb, ['Total Description', 'Year'], 'stats', processed_path)

