import pandas as pd
import matplotlib.pyplot as plt

def load_simulated_data(path="data/simulated_entsoe_data.csv"):
    df = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
    return df

if __name__ == "__main__":
    df = load_simulated_data()
    print(df.head())

    # Visualisierung
    df.plot(figsize=(12, 4), title="Simulierter Lastverlauf (MW)")
    plt.xlabel("Datum")
    plt.ylabel("Last [MW]")
    plt.tight_layout()
    plt.show()
