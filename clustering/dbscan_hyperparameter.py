"""
This python program searches for the best parameters for the non-parametric density-based DBSCAN clustering algorithm.
"""


import sys
import time
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


# This function implements the silhouette_score scoring for the parameter searching.
def silhouette_scorer(model, data):
    clusters = model.fit_predict(data)
    if 1 < len(np.unique(clusters)) < len(data):
        sil_score = silhouette_score(data, clusters)
    else:
        sil_score = -1
    return sil_score


if __name__ == '__main__':
    # read data
    data = pd.read_pickle('../data/clustering_input/clustering_df_categories.pkl')

    # prepare data for clustering (store and then remove id)
    user_id = data['id']
    dates = data['date']
    data.drop(columns=['id', 'date'], inplace=True)

    # --------------- min_samples parameter (vm) --------------- #
    min_samples = [20, 30, 40, 50]
    print("Performing search for min_samples ... ")
    start = time.time()
    results = []
    for min_sample in min_samples:
        print("for parameter:", min_sample)
        model = DBSCAN(eps=0.5, min_samples=min_sample)  # default eps
        score = silhouette_scorer(model, data)
        results.append((min_sample, score))
    print("finished after", time.time() - start)

    best_params, best_score = max(results, key=lambda x: x[1])
    print("Best parameter value:", best_params)
    print("Best silhouette score:", best_score)

    # --------------- eps parameter (locally) --------------- #
    # calculate the average distance between each point in the dataset and its min_samples nearest neighbors
    neighbors = NearestNeighbors(n_neighbors=50)
    neighbors_fit = neighbors.fit(data)
    distances, indices = neighbors_fit.kneighbors(data)
    # visualize the distances
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    plt.plot(distances)
    plt.show()

    # eps hyperparameter search (vm)
    eps = [0.08, 0.1, 0.12]
    print("Performing search for eps ... ")
    start = time.time()
    results = []
    for ep in eps:
        print("for parameter:", ep)
        model = DBSCAN(eps=ep, min_samples=50)
        score = silhouette_scorer(model, data)
        results.append((ep, score))
    print("finished after", time.time() - start)

    best_params, best_score = max(results, key=lambda x: x[1])
    print("Best parameter value:", best_params)
    print("Best silhouette score:", best_score)
