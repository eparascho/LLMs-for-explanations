"""
This python program performs the non-parametric agglomerative clustering algorithm.
"""

import sys
import time
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


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

    # perform agglomerative clustering
    start = time.time()
    print("Clustering with Agglomerative ... ")
    agglomerative = AgglomerativeClustering()
    results = perform_clustering(agglomerative, data, user_id)
    print("Agglomerative finished after", time.time() - start)
    results.to_csv('../data/clustering_results/agglomerative_results.csv')
