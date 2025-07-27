# ml_pipeline.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost nicht installiert – 'xgb' Modell deaktiviert.")

# 1. Daten und Feature Engineering
def load_and_prepare_data(path="data/simulated_entsoe_data.csv"):
    df = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["lag_1"] = df["load_MW"].shift(1)
    df["lag_24"] = df["load_MW"].shift(24)
    df.dropna(inplace=True)

    features = ["hour", "day_of_week", "is_weekend", "lag_1", "lag_24"]
    X = df[features]
    y = df["load_MW"]
    return train_test_split(X, y, test_size=0.2, shuffle=False), df

# 2. Modelle definieren
def train_linear_regression(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train, X_test, y_test):
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost ist nicht installiert.")
    model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
    model.fit(X_train, y_train)
    return model

# 3. Visualisierung
def plot_predictions(y_test, preds, model_name="Model"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test.index, y=y_test, name="True Load", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=y_test.index, y=preds, name=f"{model_name} Prediction", line=dict(color='red')))
    fig.update_layout(title=f"{model_name} Load Forecast", xaxis_title="Time", yaxis_title="Load (MW)")
    fig.show()

# 4. Main-Aufruf mit CLI
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='rf', help='rf | lr | xgb')
    args = parser.parse_args()

    (X_train, X_test, y_train, y_test), df = load_and_prepare_data()

    if args.model == 'rf':
        model = train_random_forest(X_train, y_train, X_test, y_test)
    elif args.model == 'lr':
        model = train_linear_regression(X_train, y_train, X_test, y_test)
    elif args.model == 'xgb':
        model = train_xgboost(X_train, y_train, X_test, y_test)
    else:
        raise ValueError("Model not supported. Choose from: rf, lr, xgb")

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"\n📊 {args.model.upper()} Ergebnis:")
    print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}")

    # Plot (interaktiv)
    plot_predictions(y_test, preds, model_name=args.model.upper())

if __name__ == "__main__":
    main()
