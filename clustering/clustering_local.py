"""
    This python program performs and evaluates different baseline clustering algorithms. These algorithms are the following:
    1. Parametric square-error k-means
    2. Parametric multi-view spectral
    3. Non-parametric density-based DBSCAN
    4. Non-parametric hierarchical density-based HDBSCAN
    5. Non-parametric agglomerative
"""
import time
import pandas as pd
from sklearn.cluster import KMeans
from functions.clustering import elbow_method, silhouette_method, perform_clustering, evaluation_metrics


if __name__ == '__main__':
    # read data
    data = pd.read_pickle('../data/clustering_df.pkl')

    # prepare data for clustering (store and then remove id)
    user_id = data['id']
    data.drop(columns=['id'], inplace=True)

    # ------------- k-means ------------- #

    # find optimal k for k-means with the elbow method
    # elbow_method('kmeans', data)

    # find optimal k for k-means with the silhouette method
    # silhouette_method('kmeans', data)

    # perform k-means clustering
    # start = time.time()
    # print("Clustering with K-means ... ")
    # kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto")
    # y, results = perform_clustering(kmeans, data, user_id)
    # evaluation_metrics(data, y)
    # print("K-means finished after", time.time() - start)
    # results.to_csv('../data/clustering_results/kmeans_results.csv')

    # ------------- spectral ------------- #

    # find optimal k for spectral with the elbow method
    elbow_method('spectral', data)

    # find optimal k for spectral with the silhouette method
    silhouette_method('spectral', data)

    # # perform spectral clustering
    # start = time.time()
    # print("Clustering with Spectral ... ")
    # spectral = SpectralClustering(n_clusters=2, assign_labels='cluster_qr', random_state=0)
    # y, results = perform_clustering(spectral, data)
    # evaluation_metrics(data, y)
    # print("Spectral finished after", time.time() - start)
    # results.to_csv('../data/clustering_results/spectral_results.csv')
