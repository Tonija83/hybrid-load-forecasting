# evaluate_models.py

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Daten laden und vorbereiten
df = pd.read_csv("data/simulated_entsoe_data.csv", parse_dates=["datetime"], index_col="datetime")
df["hour"] = df.index.hour
df["day_of_week"] = df.index.dayofweek
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
df["lag_1"] = df["load_MW"].shift(1)
df["lag_24"] = df["load_MW"].shift(24)
df.dropna(inplace=True)

X = df[["hour", "day_of_week", "is_weekend", "lag_1", "lag_24"]]
y = df["load_MW"]

# Cross-Validation vorbereiten
tscv = TimeSeriesSplit(n_splits=5)
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
}

results = []

# Modelle evaluieren
for name, model in models.items():
    print(f"Evaluating: {name}")
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        results.append({
            "Model": name,
            "Fold": fold + 1,
            "RMSE": mean_squared_error(y_test, preds, squared=False),
            "MAE": mean_absolute_error(y_test, preds),
            "R2": r2_score(y_test, preds)
        })

# Ergebnisse analysieren
results_df = pd.DataFrame(results)
print(results_df.groupby("Model").mean().round(2))

# Visualisierung
plt.figure(figsize=(10, 6))
sns.boxplot(data=results_df, x="Model", y="RMSE")
plt.title("RMSE je Modell (TimeSeriesSplit)")
plt.tight_layout()
plt.savefig("figures/model_comparison_rmse.png")
plt.show()
