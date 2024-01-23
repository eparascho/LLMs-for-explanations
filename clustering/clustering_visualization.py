"""
This file contains the code for visualizing the clustering results.
"""


import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # read raw data
    data = pd.read_pickle('../data/clustering_df.pkl')
    column_names = data.columns.values.tolist()  # raw data column names
    column_names.append('cluster')  # cluster

    # read and prepare hdbscan results
    hdbscan = pd.read_csv('../data/clustering_results/hdbscan_results.csv')
    hdbscan.drop(hdbscan.columns[0], axis=1, inplace=True)  # drop the first column
    hdbscan.columns = column_names  # rename columns

    boxplot_data = [hdbscan[hdbscan['cluster'] == cluster]['exercise_duration'] for cluster in hdbscan['cluster'].unique()]
    plt.figure(figsize=(8, 6))

    # Colors for each box
    colors = ['#ffcc99', '#ccccff', '#ffffcc', '#ccffe6', '#b3e6ff']
    boxplot_elements = plt.boxplot(boxplot_data, patch_artist=True, labels=hdbscan['cluster'].unique())
    for patch, color in zip(boxplot_elements['boxes'], colors):
        patch.set_facecolor(color)

    plt.title('Box Plot by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Feature Values')
    plt.show()
