import pandas as pd
from pathlib import Path
import sys

source_path = Path('data/processed/final')
source_path_region = Path('data/processed/final/region')
source_path_suburb = Path('data/processed/final/suburb')

#def file_validation(df):


region_names = [f.name for f in source_path_region.iterdir() if f.is_file()]
for f in region_names:
    region_df = pd.read_csv(source_path_region / f)
    print(region_df.head())

    sys.exit()
