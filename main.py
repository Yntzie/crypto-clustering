from src.data_loader import load_crypto_data
from src.features import build_features
from src.clustering import (
    elbow_method,
    silhouette_analysis,
    kmeans_final,
    hierarchical_clustering
)
from src.visualization import plot_pca


data = load_crypto_data(period="1y")
feature_df, X_scaled = build_features(data)

# STEP 1
elbow_method(X_scaled)
silhouette_analysis(X_scaled)

# STEP 2
feature_df = kmeans_final(X_scaled, feature_df, k=3)
feature_df = hierarchical_clustering(X_scaled, feature_df, k=3)

print(feature_df[['symbol', 'kmeans_cluster', 'hier_cluster']])

# PCA Visualization (K-Means)
plot_pca(X_scaled, feature_df, "kmeans_cluster")


output_path = "cluster_result.csv"
feature_df.to_csv(output_path, index=False)
print(f"[INFO] Cluster result saved to {output_path}")
