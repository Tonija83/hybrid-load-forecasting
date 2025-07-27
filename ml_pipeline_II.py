import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# 1. Daten laden
df = pd.read_csv("data/simulated_entsoe_data.csv", parse_dates=["datetime"], index_col="datetime")
series = df["load_MW"].values.reshape(-1, 1)

# 2. Skalieren
scaler = MinMaxScaler()
scaled = scaler.fit_transform(series)

# 3. Daten vorbereiten (Sequenzen erzeugen)
def prepare_data(series, n_steps):
    X, y = [], []
    for i in range(len(series) - n_steps):
        X.append(series[i:i+n_steps])
        y.append(series[i+n_steps])
    X = np.array(X)
    y = np.array(y)
    return X.reshape((X.shape[0], X.shape[1], 1)), y

n_steps = 24
X, y = prepare_data(scaled, n_steps)

# 4. Split in Training und Test
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 5. Modell erstellen
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

model = create_lstm_model((n_steps, 1))

# 6. Training mit EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                    validation_split=0.1, callbacks=[es], verbose=1)

# 7. Vorhersage
y_pred = model.predict(X_test)
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# 8. Plot
plt.figure(figsize=(12, 5))
plt.plot(y_test_inv[:100], label="True Load", linestyle='--')
plt.plot(y_pred_inv[:100], label="LSTM Forecast")
plt.legend()
plt.title("LSTM Forecast vs Actual Load")
plt.xlabel("Stunde")
plt.ylabel("Load (MW)")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/lstm_forecast_real.png")
plt.show()
