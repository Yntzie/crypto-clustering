# Crypto Clustering using Machine Learning

## 📌 Project Description
This project performs clustering analysis on cryptocurrency data using
K-Means and Hierarchical Clustering to identify groups of assets with similar
market behavior.

## 📊 Dataset
- Assets: BTC, ETH, BNB, SOL, ADA, XRP, DOGE, AVAX
- Timeframe: 6 months – 1 year
- Source: Yahoo Finance

## ⚙️ Features Used
- Mean Return
- Volatility
- Maximum Drawdown
- Average Volume
- RSI
- Price Range

## 🧠 Methods
- Feature Scaling (StandardScaler)
- K-Means Clustering
- Hierarchical Clustering (Ward linkage)
- PCA for visualization
- Evaluation using Silhouette Score

## 📈 Results
- Optimal number of clusters: K = 4
- Bitcoin consistently forms its own cluster
- Altcoins grouped based on risk and volatility

## 📂 Output
- `cluster_result.csv`: clustering result
- PCA visualization
- Dendrogram visualization

## ▶️ How to Run
```bash
pip install -r requirements.txt
python main.py
