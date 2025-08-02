
import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    
    df["AMT_OVER_220"] = (df["TX_AMOUNT"] > 220).astype(int)

    
    df["TX_HOUR"]    = df["TX_DATETIME"].dt.hour
    df["TX_DOW"]     = df["TX_DATETIME"].dt.dayofweek
    df["IS_WEEKEND"]= df["TX_DOW"].isin([5,6]).astype(int)

    
    df = df.sort_values("TX_DATETIME")

    
    df["TERMINAL_RISK_28D"] = 0.0
    for term_id, grp in df.groupby("TERMINAL_ID", sort=False):
        grp = grp.sort_values("TX_DATETIME")
        
        rolling_risk = grp.rolling("28D", on="TX_DATETIME")["TX_FRAUD"].mean()
        df.loc[grp.index, "TERMINAL_RISK_28D"] = rolling_risk.values

    
    df["CUST_MED_14D"] = 0.0
    df["CUST_SPIKE"]   = 0.0
    for cust_id, grp in df.groupby("CUSTOMER_ID", sort=False):
        grp = grp.sort_values("TX_DATETIME")
        median_14d = grp.rolling("14D", on="TX_DATETIME")["TX_AMOUNT"].median()
        spike_14d  = grp["TX_AMOUNT"] / (median_14d + 1e-3)
        df.loc[grp.index, "CUST_MED_14D"] = median_14d.values
        df.loc[grp.index, "CUST_SPIKE"]   = spike_14d.values

    
    df["CUSTOMER_ID"] = df["CUSTOMER_ID"].astype("category").cat.codes
    df["TERMINAL_ID"] = df["TERMINAL_ID"].astype("category").cat.codes

    return df
