import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(period).mean() / loss.rolling(period).mean()
    return 100 - (100 / (1 + rs))


def build_features(data_dict):
    features = []

    for symbol, df in data_dict.items():
        df = df.dropna()

        returns = df['Close'].pct_change()

        mean_return = returns.mean().item()
        volatility = returns.std().item()

        cum_return = (1 + returns).cumprod()
        max_drawdown = (cum_return / cum_return.cummax() - 1).min().item()

        avg_volume = df['Volume'].mean().item()
        rsi = compute_rsi(df['Close']).mean().item()
        price_range = (df['High'] - df['Low']).mean().item()

        features.append([
            symbol,
            mean_return,
            volatility,
            max_drawdown,
            avg_volume,
            rsi,
            price_range
        ])

    feature_df = pd.DataFrame(features, columns=[
        "symbol",
        "mean_return",
        "volatility",
        "max_drawdown",
        "avg_volume",
        "rsi",
        "price_range"
    ])

    X = feature_df.drop("symbol", axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return feature_df, X_scaled

