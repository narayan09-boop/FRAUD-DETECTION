from src.load_data import load_all_data
from src.eda import eda
from src.features import add_features
from src.split_data import split_timewise
from src.model import train_and_evaluate
import joblib

def run_pipeline():
    print("🔁 Loading data from .pkl files...")
    df = load_all_data()

    print("📊 Running Exploratory Data Analysis...")
    eda(df)

    print("⚙️ Adding Features...")
    df = add_features(df)
    print("✅ Features added!")

    print("✂️ Splitting into Train/Test sets (based on time)...")
    X_train, y_train, X_test, y_test = split_timewise(df)

    print("🧠 Training model & evaluating...")
    model = train_and_evaluate(X_train, y_train, X_test, y_test)

    print("💾 Saving trained model to fraud_model.pkl...")
    joblib.dump(model, "fraud_model.pkl")
    print("🎉 Model saved successfully!")

if __name__ == "__main__":
    run_pipeline()
