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
    data['cluster'] = data['cluster'].astype(str)
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
    data['cluster'] = data['cluster'].astype(str)
    plt.figure(figsize=(8, 6))
    if col == 'mood':
        ax = sns.barplot(x='cluster', y=col, data=data, palette=['#ffcc99', '#ccccff', '#ffffcc', '#ccffe6', '#b3e6ff', '#ff704d',
                                                                 '#ffccff', '#80ffaa', '#6666ff', '#c1c1a4', '#ffb3ff', '#99ffeb', '#dfbf9f'])
    else:
        ax = sns.barplot(x='cluster', y=col, data=data, palette=['#ffcc99', '#ccccff', '#ffffcc', '#ccffe6', '#b3e6ff'])
    pairs = [(str(i), str(j)) for i in data['cluster'].unique() for j in data['cluster'].unique() if i < j]
    annotator = Annotator(ax, pairs, data=data, x='cluster', y=col)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
    annotator.apply_and_annotate()
    plt.xlabel('Cluster')
    plt.ylabel(col)
    filename = '../images/' + where + '/kmeans_categories_' + col + '.png'
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
    cols = list(data.columns[20:])  # 39 for full features | 20 for categories features
    for col in cols:
        # keep only the data that does not contain NaN values in this specific feature
        visualize_data = data.dropna(subset=[col])
        print("Feature", col, "has", len(visualize_data), "non-nan values")
        try:
            # for the arithmetic features: visualize the clean data with a boxplot
            boxplot(visualize_data, col, 'clustering_labeling')
        except Exception:
            try:
                # for the categorical features: visualize the clean data with a bar plot
                bar_plot(visualize_data, col, 'clustering_labeling')
            except Exception as e2:
                print("Feature", col, "could not be visualized, because", e2)


if __name__ == '__main__':
    # visualize the training features along with the clustering results
    data = pd.read_pickle('../data/labeling_visualizations/kmeans_categories_labeling.pkl')
    # visualize_training(data)

    # visualize the labeling features along with the clustering results
    visualize_labeling(data)
