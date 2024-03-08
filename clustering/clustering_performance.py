"""
This python program evaluates the performance of different clustering algorithms.
"""


import pandas as pd
from functions.clustering import evaluation_metrics


if __name__ == '__main__':
    # evaluate the model
    model = pd.read_csv('../data/clustering_results/hdbscan_results_categories.csv')
    # drop id and date columns
    model = model.iloc[:, 2:]
    evaluation_metrics(model)
