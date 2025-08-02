def split_timewise(df):
    df = df.sort_values("TX_DATETIME")

    train = df[df["TX_DATETIME"] < "2018-08-01"]
    test  = df[df["TX_DATETIME"] >= "2018-08-01"]

   
    features = [
        "TX_AMOUNT", "AMT_OVER_220", "TX_HOUR", "TX_DOW", "IS_WEEKEND",
        "TERMINAL_RISK_28D", "CUST_SPIKE", "CUSTOMER_ID", "TERMINAL_ID"
    ]

    X_train, y_train = train[features], train["TX_FRAUD"]
    X_test, y_test   = test[features],  test["TX_FRAUD"]

    return X_train, y_train, X_test, y_test
