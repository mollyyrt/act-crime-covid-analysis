import pandas as pd
import numpy as np

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


def create_suburb_df(f, suburb_path):
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
    df_all = df.groupby(group_list, as_index=False).sum()
    df_all['Crime'] = 'All Crime'
    df_total = pd.concat([df_all, df])
    return df_total


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


def calculate_rate(df):
    """
    Calculates rate of measure per 1,000 population from a count

    Args:
        df: (pd.Dataframe) Dataframe containing 'Number' and 'Population'columns
    Returns
        df: (pd.Dataframe) Dataframe including 'Rate' column

    """
    df['Thousand Pop'] = df['Population']/1000
    df['Rate'] = df['Number'] / df['Thousand Pop'].mask(df['Thousand Pop'] == 0, np.nan)
    df['Rate'] = df['Rate'].astype(float).fillna(0)
    df = df.drop(columns=['Thousand Pop'])
    return df

def datetime(df):
    """
    Get datetime column from 'Year' and 'Quarter' columns

    Args:
        df: (pd.Dataframe) dataframe containing 'Year' and 'Quarter' columns

    Returns:
        df: (pd.Dataframe) Updated dataframe including pd.datetime ('Date') column
    """
    if 'Quarter' in df.columns:
        y_q = df['Year'].astype(str) + '-' + df['Quarter']
        df['Date'] = pd.PeriodIndex(y_q, freq='Q').to_timestamp()
    return df

def add_covid_col(df):
    """
    Adds 'Covid' column to crime data, specifying time period in relation to covid 
    
    Args:
        df: (pd.Dataframe) Crime data containing 'Year' and 'Quarter' columns
    Returns: 
            df: (pd.Dataframe) Updated dataframe
    """
    if 'Quarter' in df.columns:
        df['Covid'] = np.where(((df['Year'] == 2021) & (df['Quarter'] != 'Q4')) | 
                            ((df['Year'] == 2020) & (df['Quarter'] != 'Q1')), 'During', 
                            np.where(((df['Year'] == 2020) & (df['Quarter'] == 'Q1')) | 
                                        (df['Year'] < 2020), 'Pre', 'Post'))
    else:
        df['Covid'] = np.where((df['Year'] == 2021) | (df['Year'] == 2020), 'During', 
                               np.where(df['Year'] < 2020, 'Pre', 'Post'))
        
    df['Covid'] = pd.Categorical(df['Covid'], categories=['Pre', 'During', 'Post'], ordered=True)
    return df

def save_region_suburb(suburb_df, group_cols, title, target_dir):
    """
    Creates region and ACT level data from suburb level dataframe
    Includes measurement rates per 1,000 population, covid period flags and datetime information
    Saves dataframe to target directory
    Args:
        df: (pd.Dataframe) Dataframe containing 'Suburb', 'Region', 'Number', 'Population' and group_cols columns
        group_cols: (list(str)) Specifies df columns to be used for grouping
        title: (str) Save file name prefix
        target_dir: (pathlib.Path) File save path
    """
   
    region_df = suburb_df.groupby(['Region'] + group_cols).sum().reset_index().drop(columns=['Suburb'])
    act_df = suburb_df.groupby(group_cols).sum().reset_index().drop(columns=['Suburb', 'Region'])

    df_list = [suburb_df, region_df, act_df]
    df_list = [df.pipe(calculate_rate) for df in df_list]
    df_list = [df.pipe(datetime) for df in df_list]
    df_list = [df.pipe(add_covid_col) for df in df_list]

    f_names = [title + '_suburb.csv', title + '_region.csv', title + '_act.csv']
    for i in range(3):
        df_list[i].to_csv(target_dir / f_names[i], index=False)
