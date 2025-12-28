import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

def plot_pca(X_scaled, feature_df, cluster_col):
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(
        components,
        columns=["PC1", "PC2"]
    )

    pca_df["cluster"] = feature_df[cluster_col].values
    pca_df["symbol"] = feature_df["symbol"].values

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        pca_df["PC1"],
        pca_df["PC2"],
        c=pca_df["cluster"],
        cmap="tab10",
        s=100
    )

    for i, symbol in enumerate(pca_df["symbol"]):
        plt.text(
            pca_df["PC1"][i] + 0.02,
            pca_df["PC2"][i] + 0.02,
            symbol,
            fontsize=9
        )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Visualization of Crypto Clusters")
    plt.colorbar(scatter, label="Cluster")
    plt.grid(True)
    plt.show()
