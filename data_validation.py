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


