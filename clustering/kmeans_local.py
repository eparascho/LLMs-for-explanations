"""
This python program performs the parametric square-error k-means clustering algorithm.
"""


import time
import pandas as pd
from sklearn.cluster import KMeans
from functions.clustering import elbow_method, silhouette_method, perform_clustering


if __name__ == '__main__':
    # read data
    data = pd.read_pickle('../data/clustering_input/clustering_df_categories.pkl')

    # prepare data for clustering (store and then remove id)
    user_id = data['id']
    dates = data['date']
    data.drop(columns=['id', 'date'], inplace=True)

    # find optimal k for k-means with the elbow method
    # elbow_method('kmeans', 'categories', data)

    # find optimal k for k-means with the silhouette method
    # silhouette_method('kmeans', 'categories', data)

    # perform k-means clustering
    start = time.time()
    print("Clustering with K-means ... ")
    kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto")
    metadata = pd.concat([user_id, dates], axis=1)
    results = perform_clustering(kmeans, data, metadata)
    print("K-means finished after", time.time() - start)
    results.to_csv('../data/clustering_results/kmeans_results_categories.csv', index=False)
