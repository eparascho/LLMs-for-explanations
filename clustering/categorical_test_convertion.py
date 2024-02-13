"""
This file tests the statistical significances of the categorical features id first converted to numerical features.
"""


import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from statannotations.Annotator import Annotator


if __name__ == '__main__':
    data = pd.read_pickle('../data/labeling_visualizations/kmeans_clean_labeling.pkl')

    # categorical features
    test_categorical_features = ['bmi', 'extraversion', 'self_determination']

    for categorical in test_categorical_features:
        # keep only the data that does not contain NaN values in this specific feature
        visualize_data = data.dropna(subset=[categorical])
        visualize_data['cluster'] = visualize_data['cluster'].astype(str)

        # convert to numerical
        if categorical == 'extraversion':
            visualize_data['extraversion'] = visualize_data['extraversion'].map({'Below average': 1, 'Average': 2, 'Above average': 3})
        elif categorical == 'bmi':
            visualize_data['bmi'] = visualize_data['bmi'].map({'Underweight': 1, 'Normal': 2, 'Overweight': 3, 'Obese': 4})
        elif categorical == 'self_determination':
            visualize_data['self_determination'] = visualize_data['self_determination'].map({'amotivation': 1, 'external_regulation': 2, 'introjected_regulation': 3,
                                                                                             'identified_regulation': 4, 'intrinsic_regulation': 5})
        # visualize with bar plot
        plt.figure(figsize=(8, 6))
        ax = sns.boxplot(x='cluster', y=categorical, data=visualize_data, palette=['#ffcc99', '#ccccff', '#ffffcc', '#ccffe6', '#b3e6ff'])
        pairs = [(str(i), str(j)) for i in visualize_data['cluster'].unique() for j in visualize_data['cluster'].unique() if i < j]
        annotator = Annotator(ax, pairs, data=visualize_data, x='cluster', y=categorical)
        annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
        annotator.apply_and_annotate()

        plt.xlabel('Cluster')
        plt.ylabel(categorical)
        filename = '../images/testing_features_convertions/' + categorical + '.png'
        plt.savefig(filename)
