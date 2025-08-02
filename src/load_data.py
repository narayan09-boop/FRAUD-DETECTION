import pandas as pd
import glob

def load_all_data(path="data/*.pkl"):
    files = sorted(glob.glob(path))
    all_data = []
    for f in files:                          
        df = pd.read_pickle(f)
        all_data.append(df)
    df = pd.concat(all_data, ignore_index=True)
    df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])
    return df
