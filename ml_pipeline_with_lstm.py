import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lstm_model import create_lstm_model, prepare_lstm_data

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

def train_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_linear_regression(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train, X_test, y_test):
    model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
    model.fit(X_train, y_train)
    return model

def train_lstm(df, target_column="load_MW", n_steps=24):
    series = df[target_column].values
    X, y = prepare_lstm_data(series, n_steps)
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    model = create_lstm_model((n_steps, 1))
    model.fit(X_train, y_train, epochs=10, verbose=0)

    preds = model.predict(X_test).flatten()
    rmse = np.sqrt(np.mean((preds - y_test)**2))
    print(f"[LSTM] RMSE: {rmse:.2f}")
    return model, preds, y_test, X_test

def plot_predictions(y_test, preds, model_name="Model"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test.index, y=y_test, name="True Load", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=y_test.index, y=preds, name=f"{model_name} Prediction", line=dict(color='red')))
    fig.update_layout(title=f"{model_name} Load Forecast", xaxis_title="Time", yaxis_title="Load (MW)")
    fig.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='rf', help='rf | lr | xgb | lstm')
    args = parser.parse_args()

    if args.model == 'rf':
        model = train_random_forest(X_train, y_train, X_test, y_test)
        preds = model.predict(X_test)
        plot_predictions(y_test, preds, model_name="Random Forest")

    elif args.model == 'lr':
        model = train_linear_regression(X_train, y_train, X_test, y_test)
        preds = model.predict(X_test)
        plot_predictions(y_test, preds, model_name="Linear Regression")

    elif args.model == 'xgb':
        model = train_xgboost(X_train, y_train, X_test, y_test)
        preds = model.predict(X_test)
        plot_predictions(y_test, preds, model_name="XGBoost")

    elif args.model == 'lstm':
        model, preds, true, _ = train_lstm(df)
        idx = df.index[-len(true):]
        plot_predictions(pd.Series(true, index=idx), pd.Series(preds, index=idx), model_name="LSTM")

    else:
        raise ValueError("Model not supported")

if __name__ == "__main__":
    main()
