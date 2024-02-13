"""
This file contains the code for visualizing the clustering results.
"""

import warnings
import pandas as pd
from functions.visualizations import boxplot, bar_plot
warnings.filterwarnings('ignore')


"""
This function visualizes the training features compared to the assigned cluster from the clustering algorithm.
"""
def visualize_training(data):
    cols = list(data.columns[2:38])
    for col in cols:
        boxplot(data, col, 'clustering_training', 'hdbscan')


"""
This function visualizes the labeling features compared to the assigned cluster from the clustering algorithm.
"""
def visualize_labeling(data):
    # numerical features
    numerical_features = ['nightly_temperature', 'nremhr', 'spo2', 'rmssd', 'full_sleep_breathing_rate', 'deep_sleep_breathing_rate', 'light_sleep_breathing_rate',
    'rem_sleep_breathing_rate', 'stress_score', 'responsiveness_points', 'wrist_temperature', 'vo2max', 'oxygen_variation', 'scl_avg', 'mindfulness_start_heart_rate',
    'mindfulness_end_heart_rate', 'resting_heart_rate', 'mood_value', 'bpm']

    # categorical features
    categorical_features = ['ecg', 'heart_rate_alert', 'gender', 'age', 'bmi', 'extraversion', 'agreeableness', 'conscientiousness', 'stability', 'intellect', 'self_determination',
    'positive_affect_score', 'negative_affect_score', 'stai_stress', 'ttm_stage', 'dramatic_relief_category', 'environmental_reevaluation_category', 'self_reevaluation_category',
    'social_liberation_category', 'reinforcement_management_category', 'self_liberation_category', 'mood']

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
    data = pd.read_pickle('../data/labeling_visualizations/kmeans_full_labeling.pkl')

    # visualize the training features along with the clustering results
    # visualize_training(data)

    # visualize the labeling features along with the clustering results
    visualize_labeling(data)
