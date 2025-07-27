# ml_pipeline.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. Daten laden
df = pd.read_csv("data/simulated_entsoe_data.csv", parse_dates=["datetime"], index_col="datetime")

# 2. Feature Engineering
df["hour"] = df.index.hour
df["day_of_week"] = df.index.dayofweek
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
df["lag_1"] = df["load_MW"].shift(1)
df["lag_24"] = df["load_MW"].shift(24)
df.dropna(inplace=True)

# 3. Features und Ziel definieren
features = ["hour", "day_of_week", "is_weekend", "lag_1", "lag_24"]
X = df[features]
y = df["load_MW"]

# 4. Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 5. Modelle definieren
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
}

results = {}

# 6. Training und Auswertung
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    results[name] = {"rmse": rmse, "mae": mae}

    # 7. Visualisierung
    plt.figure(figsize=(12, 4))
    plt.plot(y_test.values[:100], label="True Load", linestyle='--')
    plt.plot(y_pred[:100], label=f"{name} Prediction")
    plt.title(f"{name} – Forecast vs. Actual")
    plt.xlabel("Stunde")
    plt.ylabel("Load (MW)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"figures/{name.replace(' ', '_').lower()}_forecast.png")
    plt.close()

# 8. Ergebnisübersicht
print("Modellvergleich:")
for name, metrics in results.items():
    print(f"{name}: RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}")
