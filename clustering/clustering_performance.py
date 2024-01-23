"""
This python program evaluates the performance of different clustering algorithms.
"""


import pandas as pd
from functions.clustering import evaluation_metrics


if __name__ == '__main__':
    # read data
    data = pd.read_pickle('../data/clustering_df.pkl')

    # prepare data for clustering (store and then remove id)
    user_id = data['id']
    data.drop(columns=['id'], inplace=True)

    # evaluate k-means
    kmeans = pd.read_csv('../data/clustering_results/kmeans_results.csv')
    kmeans_y = kmeans.iloc[:, -1]
    evaluation_metrics(data, kmeans_y)

    # evaluate HDBSCAN
    hdbscan = pd.read_csv('../data/clustering_results/hdbscan_results.csv')
    hdbscan_y = hdbscan.iloc[:, -1]
    print(hdbscan_y.nunique())
    evaluation_metrics(data, hdbscan_y)

    # evaluate DBSCAN
    dbscan = pd.read_csv('../data/clustering_results/dbscan_results.csv')
    dbscan_y = dbscan.iloc[:, -1]
    print(dbscan_y.nunique())
    evaluation_metrics(data, dbscan_y)
