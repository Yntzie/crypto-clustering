from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

def elbow_method(X_scaled, k_range=range(2, 8)):
    inertia = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    plt.figure()
    plt.plot(list(k_range), inertia, marker='o')
    plt.xlabel("Jumlah Cluster (K)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.show()


def silhouette_analysis(X_scaled, k_range=range(2, 8)):
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        print(f"K = {k}, Silhouette Score = {score:.3f}")

def kmeans_final(X_scaled, feature_df, k=3):
    kmeans = KMeans(n_clusters=k, random_state=42)
    feature_df['kmeans_cluster'] = kmeans.fit_predict(X_scaled)
    return feature_df


def hierarchical_clustering(X_scaled, feature_df, k=3):
    linked = linkage(X_scaled, method='ward')
    feature_df['hier_cluster'] = fcluster(linked, k, criterion='maxclust')

    plt.figure(figsize=(10, 5))
    dendrogram(linked, labels=feature_df['symbol'].values)
    plt.title("Dendrogram Hierarchical Clustering")
    plt.xlabel("Crypto")
    plt.ylabel("Distance")
    plt.show()

    return feature_df
