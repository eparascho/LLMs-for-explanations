"""
This file contains the functions for the visualizations.
"""


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
    ax = sns.boxplot(x='cluster', y=col, data=data, palette=['#ffcc99', '#ccccff', '#ffffcc', '#ccffe6', '#b3e6ff'])
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
