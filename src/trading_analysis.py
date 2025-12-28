import pandas as pd

def analyze_clusters(df, cluster_col="kmeans_cluster"):
    summary = (
        df.groupby(cluster_col)
        .agg(
            mean_return=("mean_return", "mean"),
            volatility=("volatility", "mean"),
            max_drawdown=("max_drawdown", "mean"),
            avg_volume=("avg_volume", "mean"),
            rsi=("rsi", "mean")
        )
        .reset_index()
    )

    return summary

def assign_strategy(summary_df):
    df = summary_df.copy()

    # Urutkan berdasarkan volatility
    df = df.sort_values("volatility")

    strategies = ["HOLD", "SWING", "SCALPING"]

    df["strategy"] = strategies[:len(df)]

    return df
