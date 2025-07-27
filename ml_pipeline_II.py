from lstm_model import create_lstm_model, prepare_lstm_data
import tensorflow as tf

def train_lstm(df, target_column="load_MW", n_steps=24):
    series = df[target_column].values
    X, y = prepare_lstm_data(series, n_steps)
    
    # Zeitsplit
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    model = create_lstm_model((n_steps, 1))
    model.fit(X_train, y_train, epochs=10, verbose=0)

    preds = model.predict(X_test).flatten()
    true = y_test
    rmse = np.sqrt(np.mean((preds - true)**2))
    print(f"[LSTM] RMSE: {rmse:.2f}")

    return model, preds, y_test, X_test
