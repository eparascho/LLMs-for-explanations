"""
    This python program performs and evaluates different baseline clustering algorithms. These algorithms are the following:
    1. Parametric square-error k-means
    2. Parametric multi-view spectral
    3. Non-parametric density-based DBSCAN
    4. Non-parametric hierarchical density-based HDBSCAN
    5. Non-parametric agglomerative
"""

import sys
import time
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import DBSCAN, HDBSCAN, SpectralClustering, KMeans, AgglomerativeClustering


"""
This function perform the clustering based on the corresponding initialized model 
"""
def perform_clustering(model, data):
    y = model.fit_predict(data)
    model_data = data.copy()
    model_data['cluster'] = y
    model_data = pd.concat([user_id, model_data], axis=1, ignore_index=True)

    return y, model_data


"""
This function calculates and prints the clustering evaluation metrics
"""
def evaluation_metrics(data, y):
    print("The Silhouette score is:", silhouette_score(data, y))
    print("The Davies-Bouldin Index is:", davies_bouldin_score(data, y))
    print("The Calinski-Harabasz Index is:", calinski_harabasz_score(data, y))


if __name__ == '__main__':
    # read arguments from command line
    input_file = sys.argv[1]

    # load the input dataframe
    data = pd.read_pickle(input_file)

    # prepare data for clustering (store and then remove id)
    user_id = data['id']
    data.drop(columns=['id'], inplace=True)

    # perform k-means clustering
    start = time.time()
    print("Clustering with K-means ... ")
    kmeans = KMeans(n_clusters=8, random_state=0, n_init="auto")
    y, results = perform_clustering(kmeans, data)
    evaluation_metrics(data, y)
    print("K-means finished after", time.time() - start)
    results.to_csv('../data/clustering_results/kmeans_results.csv')

    # perform spectral clustering
    start = time.time()
    print("Clustering with Spectral ... ")
    spectral = SpectralClustering(n_clusters=2, assign_labels='cluster_qr', random_state=0)
    y, results = perform_clustering(spectral, data)
    evaluation_metrics(data, y)
    print("Spectral finished after", time.time() - start)
    results.to_csv('../data/clustering_results/spectral_results.csv')

    # perform dbscan clustering
    start = time.time()
    print("Clustering with DBSCAN ... ")
    dbscan = DBSCAN(eps=3, min_samples=2)
    y, results = perform_clustering(dbscan, data)
    evaluation_metrics(data, y)
    print("DBSCAN finished after", time.time() - start)
    results.to_csv('../data/clustering_results/dbscan_results.csv')

    # perform hdbscan clustering
    start = time.time()
    print("Clustering with HDBSCAN ... ")
    hdbscan = HDBSCAN(min_cluster_size=1000)
    y, results = perform_clustering(hdbscan, data)
    evaluation_metrics(data, y)
    print("HDBSCAN finished after", time.time() - start)
    results.to_csv('../data/clustering_results/hdbscan_results.csv')

    # perform agglomerative clustering
    start = time.time()
    print("Clustering with Agglomerative ... ")
    agglomerative = AgglomerativeClustering()
    y, results = perform_clustering(agglomerative, data)
    evaluation_metrics(data, y)
    print("Agglomerative finished after", time.time() - start)
    results.to_csv('../data/clustering_results/agglomerative_results.csv')
