"""
This file contains the code for visualizing the clustering results.
"""

import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator
warnings.filterwarnings('ignore')

"""
This function creates and saves locally a boxplot.
"""
def boxplot(data, col, where):
    plt.figure(figsize=(8, 6))
    ax = sns.boxplot(x='cluster', y=col, data=data, palette=['#ffcc99', '#ccccff', '#ffffcc', '#ccffe6', '#b3e6ff'])
    pairs = [(str(i), str(j)) for i in data['cluster'].unique() for j in data['cluster'].unique() if i < j]
    annotator = Annotator(ax, pairs, data=data, x='cluster', y=col)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
    annotator.apply_and_annotate()
    plt.xlabel('Cluster')
    plt.ylabel(col)
    filename = '../images/' + where + '/kmeans_categories_' + col + '.png'
    plt.savefig(filename)


"""
This function creates and saves locally a bar plot.
"""
def bar_plot(data, col, where):
    plt.figure(figsize=(8, 6))
    visualize_data = data.groupby(['cluster', col]).size().reset_index(name='count')
    if col == 'mood':
        ax = sns.barplot(x='cluster', y='count', data=visualize_data, hue=col, palette=['#ffcc99', '#ccccff', '#ffffcc', '#ccffe6', '#b3e6ff', '#ff704d',
                                                                 '#ffccff', '#80ffaa', '#6666ff', '#c1c1a4', '#ffb3ff', '#99ffeb', '#dfbf9f'])
    else:
        ax = sns.barplot(x='cluster', y='count', data=visualize_data, hue=col, palette=['#ffcc99', '#ccccff', '#ffffcc', '#ccffe6', '#b3e6ff'])
    pairs = [(str(i), str(j)) for i in data['cluster'].unique() for j in data['cluster'].unique() if i < j]
    annotator = Annotator(ax, pairs, data=visualize_data, x='cluster', y='count')
    annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
    annotator.apply_and_annotate()
    plt.xlabel('Cluster')
    plt.ylabel(col)
    filename = '../images/' + where + '/kmeans_full_' + col + '.png'
    plt.savefig(filename)


"""
This function visualizes the training features compared to the assigned cluster from the clustering algorithm.
"""
def visualize_training(data):
    cols = list(data.columns[2:38])
    for col in cols:
        boxplot(data, col, 'clustering_training')


"""
This function visualizes the labeling features compared to the assigned cluster from the clustering algorithm.
"""
def visualize_labeling(data):
    # numerical features
    numerical_features = ['nremhr', 'spo2', 'rmssd', 'full_sleep_breathing_rate', 'deep_sleep_breathing_rate', 'light_sleep_breathing_rate',
    'rem_sleep_breathing_rate', 'stress_score', 'responsiveness_points', 'wrist_temperature', 'vo2max', 'oxygen_variation', 'scl_avg', 'mindfulness_start_heart_rate',
    'mindfulness_end_heart_rate', 'resting_heart_rate', 'mood_value', 'bpm']

    # categorical features
    categorical_features = ['ecg', 'heart_rate_alert', 'gender', 'age', 'bmi', 'extraversion', 'agreeableness', 'conscientiousness', 'stability', 'intellect', 'self_determination',
    'positive_affect_score', 'negative_affect_score', 'stai_stress', 'ttm_stage', 'dramatic_relief_category', 'environmental_reevaluation_category', 'self_reevaluation_category',
    'social_liberation_category', 'reinforcement_management_category', 'self_liberation_category', 'mood']

    for numerical in numerical_features:
        # keep only the data that does not contain NaN values in this specific feature
        visualize_data = data.dropna(subset=[numerical])
        visualize_data['cluster'] = visualize_data['cluster'].astype(str)
        # for the arithmetic features: visualize the clean data with a boxplot
        boxplot(visualize_data, numerical, 'clustering_labeling')

    for categorical in categorical_features:
        # keep only the data that does not contain NaN values in this specific feature
        visualize_data = data.dropna(subset=[categorical])
        visualize_data['cluster'] = visualize_data['cluster'].astype(str)
        # for the categorical features: visualize the clean data with a bar plot
        bar_plot(visualize_data, categorical, 'clustering_labeling')

    # nightly_temperature raises error when visualizing and need special handling
    # # keep only the data that does not contain NaN values in this specific feature
    # visualize_data = data.dropna(subset=['nightly_temperature'])
    # visualize_data['cluster'] = visualize_data['cluster'].astype(str)
    # # for the arithmetic features: visualize the clean data with a boxplot
    # boxplot(visualize_data, 'nightly_temperature', 'clustering_labeling')  # manually remove the annotator


if __name__ == '__main__':
    # visualize the training features along with the clustering results
    data = pd.read_pickle('../data/labeling_visualizations/kmeans_categories_labeling.pkl')
    # visualize_training(data)

    # visualize the labeling features along with the clustering results
    visualize_labeling(data)
