"""
This file contains the functions for the visualizations.
"""


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator


"""
This function annotates a plot.
"""
def do_annotation(visualize_data, ax, col):
    pairs = [(str(i), str(j)) for i in visualize_data['cluster'].unique() for j in visualize_data['cluster'].unique() if i < j]
    annotator = Annotator(ax, pairs, data=visualize_data, x='cluster', y=col)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
    annotator.apply_and_annotate()


"""
This function creates and saves a boxplot.
"""
def boxplot(data, col, where, model, version):
    plt.figure(figsize=(8, 6))
    ax = sns.boxplot(x='cluster', y=col, data=data, palette=['#ccccff', '#ffff99'])
    if col != 'nightly_temperature':  # nightly_temperature raises error when visualizing and need special handling
        do_annotation(data, ax, col)
    plt.xlabel('Cluster')
    plt.ylabel(col)
    filename = '../../images/' + where + '/' + model + '/' + col + '_' + version + '.png'
    plt.savefig(filename)


"""
This function creates and saves a bar plot.
"""
def bar_plot(data, col, model, version):
    plt.figure(figsize=(8, 6))
    visualize_data = data.groupby(['cluster', col]).size().reset_index(name='count')
    if col in ['positive_affect_score', 'negative_affect_score', 'stai_stress', 'extraversion', 'agreeableness', 'conscientiousness', 'stability', 'intellect', 'dramatic_relief_category',
               'environmental_reevaluation_category', 'self_reevaluation_category', 'social_liberation_category', 'reinforcement_management_category', 'self_liberation_category']:
        custom_order = ['Below average', 'Average', 'Above average']
    elif col == 'bmi':
        custom_order = ['Underweight', 'Normal', 'Overweight', 'Obese']
    elif col == 'self_determination':
        custom_order = ['amotivation', 'external_regulation', 'introjected_regulation', 'identified_regulation', 'intrinsic_regulation']
    elif col == 'ttm_stage':
        custom_order = ['Precontemplation', 'Contemplation', 'Preparation', 'Action', 'Maintenance']
    else:
        custom_order = visualize_data[col].unique()
    sns.barplot(x='cluster', y='count', data=visualize_data, hue=col, hue_order=custom_order, palette=['#80d4ff', '#ff99ff', '#33ff77', '#ffad33', '#ffff4d', '#00b300', '#ff5c33', '#6666ff', '#acac86', '#00e6b8',
                                                                                    '#bf8040', '#8c8c8c', '#cc0044'])
    plt.xlabel('Cluster')
    plt.ylabel(col)
    filename = '../../images/categorical_features/' + model + '/' + col + '_' + version + '.png'
    plt.savefig(filename)


"""
This function creates and saves a histogram.
"""
def histogram(data, col):
    if col == 'responsiveness_points' or col == 'resting_heart_rate' or col == 'stress_score' or col == 'rem_sleep_breathing_rate' or col == 'full_sleep_breathing_rate' or col == 'deep_sleep_breathing_rate':
        data = data[data[col] != 0]  # remove 0s from the data
    
    # dataframe for each cluster
    c0_df = data[data['cluster'] == 0]
    c1_df = data[data['cluster'] == 1]

    # plot histograms
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data=c0_df[col], label='C0', color='#9999ff', fill=True, alpha=0.3)
    sns.kdeplot(data=c1_df[col], label='C1', color='#ffff66', fill=True, alpha=0.3)
    plt.xlabel(col)
    plt.ylabel('frequency')
    plt.legend(frameon=False)
    plt.show()


"""
This function creates and saves a grouped bar chart.
"""
def grouped_barchart(data, cols):
    data = data[cols]
    data = data.groupby('cluster').sum()
    plt.subplots()
    n_clusters = len(data.index)
    index = np.arange(n_clusters)
    bar_width = 0.2
    plt.bar(index, data[cols[1]], bar_width, color='#80d4ff', label=cols[1])
    plt.bar(index + bar_width, data[cols[2]], bar_width, color='#ff99ff', label=cols[2])
    plt.bar(index + 2 * bar_width, data[cols[3]], bar_width, color='#33ff77', label=cols[3])
    plt.bar(index + 3 * bar_width, data[cols[4]], bar_width, color='#ffad33', label=cols[4])
    plt.xlabel('Cluster')
    plt.ylabel('Total duration at each sleep stage')
    plt.xticks(data.index)
    plt.legend()
    plt.tight_layout()
    plt.show()
