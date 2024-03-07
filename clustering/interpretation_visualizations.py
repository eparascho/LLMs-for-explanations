"""
This file ...
"""


import pandas as pd
from functions.visualizations import histogram, grouped_barchart


if __name__ == '__main__':
    # read data
    data = pd.read_pickle('../data/labeling_visualizations/kmeans_clean_labeling.pkl')

    # plot histograms
    histogram(data, 'scl_avg')

    # plot grouped bar charts
    grouped_barchart(data, ['cluster', 'full_sleep_breathing_rate', 'deep_sleep_breathing_rate', 'light_sleep_breathing_rate', 'rem_sleep_breathing_rate'])
