"""
This python program performs the non-parametric hierarchical density-based HDBSCAN clustering algorithm.
"""


import sys
import time
import pandas as pd
from sklearn.cluster import HDBSCAN


"""
This function perform the clustering based on the corresponding initialized model
"""
def perform_clustering(model, data, metadata):
    y = model.fit_predict(data)
    model_data = data.copy()
    model_data['cluster'] = y
    model_data = pd.concat([metadata, model_data], axis=1)

    return model_data


if __name__ == '__main__':
    # read data
    data = pd.read_pickle('../data/clustering_input/clustering_df_categories.pkl')

    # prepare data for clustering (store and then remove id)
    user_id = data['id']
    dates = data['date']
    data.drop(columns=['id', 'date'], inplace=True)

    # perform hdbscan clustering
    start = time.time()
    print("Clustering with HDBSCAN ... ")
    hdbscan = HDBSCAN(min_cluster_size=600, min_samples=4000)
    metadata = pd.concat([user_id, dates], axis=1)
    results = perform_clustering(hdbscan, data, metadata)
    print("HDBSCAN finished after", time.time() - start)
    results.to_csv('../data/clustering_results/hdbscan_results_categories.csv', index=False)
