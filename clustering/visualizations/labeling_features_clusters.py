"""
This file contains the code for visualizing the labeling features distributions within the clusters for finding clusters' statistical difference.
"""

import warnings
import pandas as pd
from functions.visualizations import boxplot
from functions.labeling_set_preprocessing import convert_categorical
warnings.filterwarnings('ignore')


"""
This function visualizes the distribution of the training features along with the assigned cluster from the clustering algorithm.
"""
def visualize_training(data):
    cols = list(data.columns[2:38])  # position of the training features within the dataframe - manual for each version full/categories/clean
    for col in cols:
        boxplot(data, col, 'clustering_training', 'hdbscan', 'full')


"""
This function visualizes the distribution of the labeling features along with the assigned cluster from the clustering algorithm.
"""
def visualize_labeling(data):
    for col in list(data.loc[:, 'ecg':'mood'].columns):
        # keep only the data that does not contain NaN values in this specific feature
        visualize_data = data.dropna(subset=[col])
        # for the features resting_heart_rate, stress_score and responsiveness_points keep only the rows that do not contain 0 values
        if col in ['resting_heart_rate', 'stress_score', 'responsiveness_points']:
            visualize_data = visualize_data[visualize_data[col] != 0]
        visualize_data['cluster'] = visualize_data['cluster'].astype(str)
        boxplot(visualize_data, col, 'clustering_labeling', 'hdbscan', 'full')


if __name__ == '__main__':
    df = pd.read_pickle('../../data/labeling_visualizations/hdbscan_full_labeling.pkl')

    # convert all the categorical features to numerical
    df = convert_categorical(df)

    # visualize the distribution of the training features within the clustering results
    # visualize_training(data)

    # visualize the distribution of the labeling features within the clustering results
    visualize_labeling(df)
