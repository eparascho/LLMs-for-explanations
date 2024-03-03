"""
This file contains the code for visualizing the categorical features before converting into numerical for finding the statistical difference.
"""

import warnings
import pandas as pd
from functions.visualizations import boxplot, bar_plot
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
    for numerical in numerical_features:
        # keep only the data that does not contain NaN values in this specific feature
        visualize_data = data.dropna(subset=[numerical])
        # for the features resting_heart_rate, stress_score and responsiveness_points keep only the rows that do not contain 0 values
        if numerical in ['resting_heart_rate', 'stress_score', 'responsiveness_points']:
            visualize_data = visualize_data[visualize_data[numerical] != 0]
        visualize_data['cluster'] = visualize_data['cluster'].astype(str)
        # for the arithmetic features: visualize the clean data with a boxplot
        boxplot(visualize_data, numerical, 'clustering_labeling', 'hdbscan', 'full')

    for categorical in categorical_features:
        # keep only the data that does not contain NaN values in this specific feature
        visualize_data = data.dropna(subset=[categorical])
        visualize_data['cluster'] = visualize_data['cluster'].astype(str)
        # for the categorical features: visualize the clean data with a bar plot
        bar_plot(visualize_data, categorical, 'clustering_labeling', 'kmeans', 'full')


if __name__ == '__main__':
    df = pd.read_pickle('../data/labeling_visualizations/kmeans_full_labeling.pkl')

    # convert all the categorical features to numerical
    # TODO: convert_categorical_to_numerical(df)

    # visualize the distribution of the training features within the clustering results
    # visualize_training(data)

    # visualize the distribution of the labeling features within the clustering results
    visualize_labeling(df)
