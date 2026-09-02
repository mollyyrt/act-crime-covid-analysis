import pandas as pd
import numpy as np

def add_covid_col(df):
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