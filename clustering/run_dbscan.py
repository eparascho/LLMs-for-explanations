"""
This python program performs the non-parametric density-based DBSCAN clustering algorithm.
"""


import sys
import time
import pandas as pd
from sklearn.cluster import DBSCAN


"""
This function perform the clustering based on the corresponding initialized model
"""
def perform_clustering(model, data, user_id):
    y = model.fit_predict(data)
    model_data = data.copy()
    model_data['cluster'] = y
    model_data = pd.concat([user_id, model_data], axis=1, ignore_index=True)

    return model_data


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
    dbscan = DBSCAN(min_samples=80, eps=0.3)
    results = perform_clustering(dbscan, data, user_id)
    print("DBSCAN finished after", time.time() - start)
    results.to_csv('../data/clustering_results/dbscan_results.csv')
