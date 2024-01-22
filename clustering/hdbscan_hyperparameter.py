"""
This python program searches for the best parameters for the non-parametric hierarchical density-based DBSCAN clustering algorithm.
"""


import sys
import time
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV


if __name__ == '__main__':
    # read arguments from command line
    input_file = sys.argv[1]

    # load the input dataframe
    data = pd.read_pickle(input_file)

    # prepare data for clustering (store and then remove id)
    user_id = data['id']
    data.drop(columns=['id'], inplace=True)

    start = time.time()
    # initialize hdbscan
    print("Initial HDBSCAN clustering ...")
    hdbscan = HDBSCAN().fit(data)
    print("finished after", time.time() - start)

    # specify parameters and distributions to sample from
    param_dist = {'min_samples': [60, 550, 1600, 4000],
                  'min_cluster_size': [600, 5500, 16000, 40000]}

    # performing grid search
    print("Performing grid search...")
    grid_search = GridSearchCV(hdbscan, param_grid=param_dist, scoring=make_scorer('adjusted_rand_score'), cv=5)
    grid_search.fit(data)
    print("finished after", time.time() - start)

    print(f"Best Parameters {grid_search.best_params_}")
    print(f"Best Score {grid_search.best_score_}")
