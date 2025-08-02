import seaborn as sns
import matplotlib.pyplot as plt

def eda(df):
    print(df.info())
    print(df.describe())
    print(df["TX_FRAUD"].value_counts(normalize=True))
    plt.figure(figsize=(10,4))
    sns.histplot(df["TX_AMOUNT"], bins=100)
    plt.title("Transaction Amount Distribution")
    plt.savefig("amount_histogram.png")
    plt.close()