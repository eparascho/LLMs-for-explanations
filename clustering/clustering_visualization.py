"""
This file contains the code for visualizing the clustering results.
"""


import pandas as pd
import matplotlib.pyplot as plt


"""
This function creates and saves locally a boxplot.
"""
def boxplot(data, col, where):
    boxplot_data = [data[data['cluster'] == cluster][col] for cluster in data['cluster'].unique()]
    plt.figure(figsize=(8, 6))
    colors = ['#ffcc99', '#ccccff', '#ffffcc', '#ccffe6', '#b3e6ff']
    boxplot_elements = plt.boxplot(boxplot_data, patch_artist=True, labels=data['cluster'].unique())
    for patch, color in zip(boxplot_elements['boxes'], colors):
        patch.set_facecolor(color)
    plt.xlabel('Cluster')
    plt.ylabel(col)
    filename = '../images/' + where + '/kmeans_' + col + '.png'
    plt.savefig(filename)


"""
This function creates and saves locally a bar plot.
"""
def bar_plot(data, col, where):
    category_counts = data.groupby(['cluster', col]).size().unstack(fill_value=0)
    plt.figure(figsize=(8, 6))
    if col == 'mood':
        colors = ['#ffcc99', '#ccccff', '#ffffcc', '#ccffe6', '#b3e6ff', '#ff704d', '#ffccff', '#80ffaa', '#6666ff', '#c1c1a4', '#ffb3ff', '#99ffeb', '#dfbf9f']
    else:
        colors = ['#ffcc99', '#ccccff', '#ffffcc', '#ccffe6', '#b3e6ff']
    category_counts.plot(kind='bar', color=colors, figsize=(8, 6))
    plt.xlabel('Cluster')
    plt.ylabel(col)
    filename = '../images/' + where + '/kmeans_' + col + '.png'
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
    cols = list(data.columns[39:])
    for col in cols:
        # keep only the data that does not contain NaN values in this specific feature
        boxplot_data = data.dropna(subset=[col])
        print("Feature", col, "has", len(boxplot_data), "non-nan values")
        try:
            # for the arithmetic features: visualize the clean data with a boxplot
            boxplot(boxplot_data, col, 'clustering_labeling')
        except Exception:
            try:
                # for the categorical features: visualize the clean data with a bar plot
                bar_plot(boxplot_data, col, 'clustering_labeling')
            except Exception as e2:
                print("Feature", col, "could not be visualized, because", e2)


if __name__ == '__main__':
    # visualize the training features along with the HDBSCAN results
    data = pd.read_pickle('../data/labeling_visualizations/kmeans_labeling_processed.pkl')
    visualize_training(data)

    # visualize the labeling features along with the HDBSCAN results
    visualize_labeling(data)
