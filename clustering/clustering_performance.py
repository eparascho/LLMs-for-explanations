"""
This python program evaluates the performance of different clustering algorithms.
"""


import pandas as pd
from functions.clustering import evaluation_metrics_old


if __name__ == '__main__':
    # evaluate k-means
    kmeans = pd.read_csv('../data/clustering_results/kmeans_results_clean.csv')
    # drop id and date columns
    kmeans = kmeans.iloc[:, 2:]
    evaluation_metrics_old(kmeans)

    # evaluate DBSCAN
    dbscan = pd.read_csv('../data/clustering_results/dbscan_results_categories.csv')
    # drop id and date columns
    dbscan = dbscan.iloc[:, 2:]
    print(dbscan.iloc[:, -1].nunique())
    evaluation_metrics_old(dbscan)

    # evaluate HDBSCAN
    hdbscan = pd.read_csv('../data/clustering_results/hdbscan_results_categories.csv')
    # drop id and date columns
    hdbscan = hdbscan.iloc[:, 2:]
    print(hdbscan.iloc[:, -1].nunique())
    evaluation_metrics_old(hdbscan)
