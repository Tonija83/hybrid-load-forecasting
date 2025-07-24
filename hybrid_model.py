import numpy as np
from lstm_model import create_lstm_model, prepare_data
from sklearn.ensemble import RandomForestRegressor

def train_hybrid(data, n_steps=10):
    # LSTM Vorbereitung
    X_lstm, y = prepare_data(data, n_steps)

    # RF Vorbereitung (flach)
    X_rf = X_lstm.reshape((X_lstm.shape[0], n_steps))

    # LSTM Modell
    lstm_model = create_lstm_model((n_steps, 1))
    lstm_model.fit(X_lstm, y, epochs=5, verbose=0)

    # LSTM Vorhersage als Feature für RF
    lstm_preds = lstm_model.predict(X_lstm, verbose=0).flatten()

    # Kombiniertes Feature-Set für RF: Original + LSTM-Vorhersage
    X_rf_combined = np.column_stack((X_rf, lstm_preds))

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_rf_combined, y)

    return lstm_model, rf_model

if __name__ == "__main__":
    data = np.arange(100)
    n_steps = 10
    lstm_model, rf_model = train_hybrid(data, n_steps)
    print("Hybridmodell trainiert.")
