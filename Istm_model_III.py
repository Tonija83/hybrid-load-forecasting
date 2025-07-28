# lstm_model.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# ---------------------
# 1. Daten einlesen
# ---------------------
df = pd.read_csv("data/simulated_entsoe_data.csv", parse_dates=["datetime"], index_col="datetime")

# Feature Engineering
df["hour"] = df.index.hour
df["day_of_week"] = df.index.dayofweek
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
df["lag_1"] = df["load_MW"].shift(1)
df["lag_24"] = df["load_MW"].shift(24)
df.dropna(inplace=True)

# Nur Zielvariable für LSTM
target_series = df["load_MW"].values

# ---------------------
# 2. LSTM-Datenstruktur
# ---------------------
def prepare_lstm_data(series, n_steps):
    X, y = [], []
    for i in range(len(series) - n_steps):
        X.append(series[i:i + n_steps])
        y.append(series[i + n_steps])
    return np.array(X).reshape((-1, n_steps, 1)), np.array(y)

n_steps = 24
X_all, y_all = prepare_lstm_data(target_series, n_steps)

# ---------------------
# 3. TimeSeriesSplit
# ---------------------
tscv = TimeSeriesSplit(n_splits=5)
rmse_scores = []

for train_idx, test_idx in tscv.split(X_all):
    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]

    # Modell definieren
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(n_steps, 1)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    model.fit(X_train, y_train, epochs=10, verbose=0)

    # Vorhersage & Bewertung
    y_pred = model.predict(X_test, verbose=0).flatten()
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    rmse_scores.append(rmse)

# ---------------------
# 4. Ergebnisse plotten
# ---------------------
plt.figure(figsize=(8, 5))
sns.boxplot(data=rmse_scores)
plt.title("LSTM RMSE über 5 Folds")
plt.ylabel("RMSE")
plt.savefig("figures/lstm_rmse_boxplot.png")
plt.show()

# ---------------------
# 5. RMSE-Ausgabe
# ---------------------
print("Durchschnittliches RMSE:", np.mean(rmse_scores))
