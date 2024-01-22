"""
    This python program performs and evaluates different baseline clustering algorithms.
    These algorithms are the following:
    1. Non-parametric density-based HDBSCAN
    2. Non-parametric agglomerative
"""

import sys
import time
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering


if __name__ == '__main__':
    # read arguments from command line
    input_file = sys.argv[1]

    # load the input dataframe
    data = pd.read_pickle(input_file)

    # prepare data for clustering (store and then remove id)
    user_id = data['id']
    data.drop(columns=['id'], inplace=True)

    # perform dbscan clustering
    start = time.time()
    print("Clustering with DBSCAN ... ")
    dbscan = DBSCAN(eps=3, min_samples=2)
    y, results = perform_clustering(dbscan, data)
    evaluation_metrics(data, y)
    print("DBSCAN finished after", time.time() - start)
    results.to_csv('../data/clustering_results/dbscan_results.csv')

    # perform agglomerative clustering
    start = time.time()
    print("Clustering with Agglomerative ... ")
    agglomerative = AgglomerativeClustering()
    y, results = perform_clustering(agglomerative, data)
    evaluation_metrics(data, y)
    print("Agglomerative finished after", time.time() - start)
    results.to_csv('../data/clustering_results/agglomerative_results.csv')
