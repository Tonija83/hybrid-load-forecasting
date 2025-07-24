import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def prepare_data(series, n_steps):
    X, y = [], []
    for i in range(len(series) - n_steps):
        X.append(series[i:i+n_steps])
        y.append(series[i+n_steps])
    X = np.array(X)
    y = np.array(y)
    return X, y

if __name__ == "__main__":
    data = np.arange(100)
    n_steps = 10
    X, y = prepare_data(data, n_steps)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test MSE: {mse:.2f}")

    # Beispielvorhersage
    x_input = np.arange(90, 100).reshape(1, -1)
    yhat = model.predict(x_input)
    print(f"Vorhersage für Input {x_input.flatten()}: {yhat[0]:.2f}")
