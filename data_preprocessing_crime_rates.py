import pandas as pd
from pathlib import Path
import sys

def add_crime_rate(crime_name, area, source_path):
    """
    For a given crime csv file, calculates crime rates using population statistics
    Saves updated csv file to a subdirectory within source path

    Args:
    crime_name: (str) Prefix of crime file name
    area: (str) specifies data granularity ('region'/'suburb')
    source_path: (pathlib.Path) specifies source file location
    """

    if area not in ('region', 'suburb'):
        raise ValueError(f"Invalid area '{area}'. Must be 'region' or 'suburb'.")
    else:
        crime_file =  crime_name + '_' + area + '.csv'
        pop_file = 'population' + '_' + area + '.csv'
        crime_path = source_path / crime_file
        pop_path = source_path / 'final' / area / pop_file
        if not crime_path.is_file():
            raise ValueError(f'{crime_path} is not a valid file')
        elif not pop_path.is_file():
            raise ValueError(f'{pop_path} is not a valid file')
        else:
            crime_df = pd.read_csv(crime_path)
            pop_df = pd.read_csv(pop_path)

            if area == 'suburb':
                merge_cols = ['Region', 'Suburb', 'Year']
            else:
                merge_cols = ['Region', 'Year']

            crime_df = pd.merge(crime_df, pop_df, on=merge_cols, how='left')

            crime_df['Thousand_Pop'] = crime_df['Population']/1000
            crime_df['Rate'] = crime_df['Number']/crime_df['Thousand_Pop'] 
            crime_df = crime_df.drop(columns=['Thousand_Pop'])
            save_path = source_path / 'final' / area 
            save_path.mkdir(parents=True, exist_ok=True)
            crime_df.to_csv(save_path / crime_file,  index=False)



for f in ['crime', 'crime_grouped']:
    for l in ['region', 'suburb']:
        add_crime_rate(f, l, Path('data/processed'))
