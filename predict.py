import pandas as pd
import joblib
import os
from src.features import add_features 

def predict_from_user_file():
    print(" Enter full path to your .pkl file (e.g., data/transactions-2018-09-30.pkl):")
    pkl_path = input(" Path: ").strip()

    if not os.path.exists(pkl_path):
        print(" File not found. Please check the path and try again.")
        return

    print(f"\n📦 Loading data from: {pkl_path}")
    df = pd.read_pickle(pkl_path)

    # 🛠️ Generate features (important!)
    print("⚙️ Generating features using add_features()...")
    df = add_features(df)

    # Load trained model
    model = joblib.load("fraud_model.pkl")

    required_features = ["TX_AMOUNT", "AMT_OVER_220", "TX_HOUR", "TX_DOW", "IS_WEEKEND",
                         "TERMINAL_RISK_28D", "CUST_SPIKE", "CUSTOMER_ID", "TERMINAL_ID"]

    df["FRAUD_PROBABILITY"] = model.predict_proba(df[required_features])[:, 1]
    df["PREDICTED_LABEL"] = (df["FRAUD_PROBABILITY"] > 0.3).astype(int)

    print(f"\n✅ Total Transactions: {len(df)}")
    print(f"🚨 Predicted Frauds: {df['PREDICTED_LABEL'].sum()}")
    print(f"✅ Legit Transactions: {(df['PREDICTED_LABEL'] == 0).sum()}")

    out_file = "fraud_prediction_output.csv"
    df.to_csv(out_file, index=False)
    print(f"\n📁 Prediction saved to: {out_file}")

if __name__ == "__main__":
    predict_from_user_file()
