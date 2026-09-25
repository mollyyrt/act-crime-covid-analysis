import pandas as pd
from pathlib import Path


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
    elif 'Region' in df.columns:
        if int(df['Region'].value_counts().nunique()) != 1:
            print(df['Region'].value_counts())
            raise ValueError(f'Some regions have missing/extra values')



source_path = Path('data/processed')

file_names = [f.name for f in source_path.iterdir() if f.is_file()]
for f in file_names:
    df = pd.read_csv(source_path / f)
    print('\n', f)
    file_validation(df)


crime_df = pd.read_csv(source_path / 'crime_suburb.csv')
stats_df = pd.read_csv(source_path / 'stats_suburb.csv')

print(f"Number of regions: {crime_df['Region'].nunique()}")
print(f"Number of suburbs: {crime_df['Suburb'].nunique()}")
print(f"Date range: {crime_df['Year'].min()} to {crime_df['Year'].max()}")
print('Number of quarters observed in each Covid period:')
print(crime_df[['Covid']].value_counts()/crime_df[['Suburb']].nunique().values[0])

print('Available socioeconomic statistics:')
print(stats_df['Total Description'].unique())

