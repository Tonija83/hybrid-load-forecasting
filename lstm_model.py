import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def prepare_data(series, n_steps):
    X, y = [], []
    for i in range(len(series) - n_steps):
        X.append(series[i:i+n_steps])
        y.append(series[i+n_steps])
    X = np.array(X)
    y = np.array(y)
    return X.reshape((X.shape[0], X.shape[1], 1)), y

if __name__ == "__main__":
    data = np.arange(100)
    n_steps = 10
    X, y = prepare_data(data, n_steps)

    model = create_lstm_model((n_steps, 1))
    model.fit(X, y, epochs=10, verbose=1)

    x_input = np.arange(90, 100).reshape((1, n_steps, 1))
    yhat = model.predict(x_input, verbose=0)
    print(f"Vorhersage für Input {x_input.flatten()}: {yhat.flatten()[0]:.2f}")
