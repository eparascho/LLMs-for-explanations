"""
This file contains the functions for the visualizations.
"""


import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator


"""
This function creates and saves a boxplot.
"""
def do_annotation(visualize_date, ax, col, what):
    pairs = [(str(i), str(j)) for i in visualize_date['cluster'].unique() for j in visualize_date['cluster'].unique() if i < j]
    if what == 'boxplot':
        annotator = Annotator(ax, pairs, data=visualize_date, x='cluster', y=col)
    else:
        annotator = Annotator(ax, pairs, data=visualize_date, x='cluster', y='count')
    annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
    annotator.apply_and_annotate()


"""
This function creates and saves a boxplot.
"""
def boxplot(data, col, where, model, version):
    plt.figure(figsize=(8, 6))
    ax = sns.boxplot(x='cluster', y=col, data=data, palette=['#ffcc99', '#ccccff', '#ffffcc', '#ccffe6', '#b3e6ff'])
    if col != 'nightly_temperature':  # nightly_temperature raises error when visualizing and need special handling
        do_annotation(data, ax, col, 'boxplot')
    plt.xlabel('Cluster')
    plt.ylabel(col)
    filename = '../images/' + where + '/' + model + '/' + col + '_' + version + '.png'
    plt.savefig(filename)


"""
This function creates and saves a bar plot.
"""
def bar_plot(data, col, where, model, version):
    plt.figure(figsize=(8, 6))
    visualize_data = data.groupby(['cluster', col]).size().reset_index(name='count')
    if col == 'mood':
        ax = sns.barplot(x='cluster', y='count', data=visualize_data, hue=col, palette=['#ffcc99', '#ccccff', '#ffffcc', '#ccffe6', '#b3e6ff', '#ff704d',
                                                                 '#ffccff', '#80ffaa', '#6666ff', '#c1c1a4', '#ffb3ff', '#99ffeb', '#dfbf9f'])
    else:
        ax = sns.barplot(x='cluster', y='count', data=visualize_data, hue=col, palette=['#ffcc99', '#ccccff', '#ffffcc', '#ccffe6', '#b3e6ff'])
    do_annotation(visualize_data, ax, col, 'barplot')
    plt.xlabel('Cluster')
    plt.ylabel(col)
    filename = '../images/' + where + '/' + model + '/' + col + '_' + version + '.png'
    plt.savefig(filename)
