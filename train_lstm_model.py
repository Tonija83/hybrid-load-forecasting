import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from lstm_model import create_lstm_model, prepare_data  # aus deiner Datei
import numpy as np

# 1. Daten laden
df = pd.read_csv("data/simulated_entsoe_data.csv", parse_dates=["datetime"], index_col="datetime")
series = df["load_MW"].values.reshape(-1, 1)

# 2. Skalieren
scaler = MinMaxScaler()
scaled = scaler.fit_transform(series)

# 3. Sequenzen erzeugen
n_steps = 24
X, y = prepare_data(scaled, n_steps)

# 4. Train/Test-Split
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 5. Modelltraining
model = create_lstm_model((n_steps, 1))
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)

# 6. Vorhersage & Rückskalierung
y_pred = model.predict(X_test)
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# 7. Plot
plt.figure(figsize=(12, 5))
plt.plot(y_test_inv[:100], label="True Load", linestyle='--')
plt.plot(y_pred_inv[:100], label="LSTM Forecast")
plt.legend()
plt.title("LSTM Forecast vs Actual Load")
plt.tight_layout()
plt.savefig("figures/lstm_forecast_real.png")
plt.show()
