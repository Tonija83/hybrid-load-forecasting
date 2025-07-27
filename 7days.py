import pandas as pd

# Simulierte Daten für 7 Tage, stündlich
data = {
    "datetime": pd.date_range("2024-01-01", periods=7*24, freq="H"),
    "load_MW": [5000 + (i % 24) * 100 for i in range(7*24)],
}
df = pd.DataFrame(data).set_index("datetime")

# Speichern
df.to_csv("data/simulated_entsoe_data.csv")
