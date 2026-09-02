import pandas as pd
from pathlib import Path
import sys
import numpy as np

def file_validation(df):
    """
    Displays data file informatioon and raises errors for irregularities
    Args:
        df: (pd.Dataframe) Processed data
    """
    print(df.shape)
    print(df.dtypes)
    nan_cols = df.columns[df.isna().sum() > 0].tolist()
    if nan_cols:
        raise ValueError(f'Columns containing missing values: {nan_cols}')
    if df.duplicated().sum() > 0:
        raise ValueError(f'Duplicates found: \n{df[df.duplicated()]}')
    if 'Suburb' in df.columns:
        if df['Suburb'].value_counts().nunique() != 1:
            raise ValueError(f'Some suburbs have missing/extra values')
    else:
        if  int(df['Region'].value_counts().nunique()) != 1:
            print(df['Region'].value_counts())
            raise ValueError(f'Some regions have missing/extra values')


def covid_dates(df):
    """
        Adds 'Covid' column to crime data, specifying time period in relation to covid 
        Args:
            df: (pd.Dataframe) Crime data containing 'Year' and 'Quarter' columns
        Returns: 
             df: (pd.Dataframe) Updated dataframe
    """
    df['Covid'] = np.where(((df['Year'] == 2021) & (df['Quarter'] != 'Q4')) | 
                           ((df['Year'] == 2020) & (df['Quarter'] != 'Q1')), 'During', 
                           np.where(((df['Year'] == 2020) & (df['Quarter'] == 'Q1')) | 
                                    (df['Year'] < 2020), 'Pre', 'Post'))
    return df


def final_file_loop(f_name, source):
    """
        Loops over files in a subdirectory and carries out data validation
        Includes 'Covid' column to crime data, specifying time period in relation to covid 
        Args:
            f_name: (str) File name
            source: (pathlib.Path) Relative location of file directory
    """
    df = pd.read_csv(source / f_name)
    print('\n', f_name)
    file_validation(df)
    if 'crime' in f_name:
        df = covid_dates(df) 
        df.to_csv(source / f_name, index=False)



source_path_region = Path('data/processed/final/region')
source_path_suburb = Path('data/processed/final/suburb')

region_names = [f.name for f in source_path_region.iterdir() if f.is_file()]
for f in region_names:
    final_file_loop(f, source_path_region)

suburb_names = [f.name for f in source_path_suburb.iterdir() if f.is_file()]
for f in suburb_names:
    final_file_loop(f, source_path_suburb)

