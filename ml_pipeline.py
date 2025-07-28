import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import plotly.graph_objects as go
import argparse

def train_linear_regression(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train, X_test, y_test):
    model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
    model.fit(X_train, y_train)
    return model

def plot_predictions(y_test, preds, model_name="Model"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test.index, y=y_test, name="True Load", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=y_test.index, y=preds, name=f"{model_name} Prediction", line=dict(color='red')))
    fig.update_layout(title=f"{model_name} Load Forecast", xaxis_title="Time", yaxis_title="Load (MW)")
    fig.write_html(f"figures/{model_name}_forecast.html")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='rf', help='rf | lr | xgb')
    args = parser.parse_args()

    df = pd.read_csv("data/simulated_entsoe_data.csv", parse_dates=["datetime"], index_col="datetime")
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["lag_1"] = df["load_MW"].shift(1)
    df["lag_24"] = df["load_MW"].shift(24)
    df.dropna(inplace=True)

    features = ["hour", "day_of_week", "is_weekend", "lag_1", "lag_24"]
    X = df[features]
    y = df["load_MW"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    if args.model == 'rf':
        model = train_random_forest(X_train, y_train, X_test, y_test)
    elif args.model == 'lr':
        model = train_linear_regression(X_train, y_train, X_test, y_test)
    elif args.model == 'xgb':
        model = train_xgboost(X_train, y_train, X_test, y_test)
    else:
        raise ValueError("Model not supported")

    preds = model.predict(X_test)
    plot_predictions(y_test, preds, model_name=args.model)

if __name__ == "__main__":
    main()
