"""
This python program searches for the best parameters for the non-parametric hierarchical density-based HDBSCAN clustering algorithm.
"""


import sys
import time
import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score


# This function implements the silhouette_score scoring for the parameter searching.
def silhouette_scorer(model, data):
    clusters = model.fit_predict(data)
    if 1 < len(np.unique(clusters)) < len(data):
        sil_score = silhouette_score(data, clusters)
    else:
        sil_score = -1
    return sil_score


if __name__ == '__main__':
    # read arguments from command line
    input_file = sys.argv[1]

    # load the input dataframe
    data = pd.read_pickle(input_file)

    # prepare data for clustering (store and then remove id)
    user_id = data['id']
    data.drop(columns=['id'], inplace=True)

    # specify parameters and distributions to sample from
    param_dist = {'min_samples': [60, 550, 1600, 4000],
                  'min_cluster_size': [600, 5500, 16000, 40000]}

    # performing grid search
    results = []
    start = time.time()
    print("Performing grid search")
    for params in ParameterGrid(param_dist):
        print("for parameters:", params)
        model = HDBSCAN(**params)
        score = silhouette_scorer(model, data)
        results.append((params, score))
    print("finished after", time.time() - start)

    best_params, best_score = max(results, key=lambda x: x[1])
    print("Best parameters:", best_params)
    print("Best silhouette score:", best_score)
