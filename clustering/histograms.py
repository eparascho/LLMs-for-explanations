"""
This file ...
"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # read data
    data = pd.read_pickle('../data/labeling_visualizations/kmeans_clean_labeling.pkl')

    # dataframe for c1 and c2
    c0_df = data[data['cluster'] == 0]
    c1_df = data[data['cluster'] == 1]
    c2_df = data[data['cluster'] == 2]
    c3_df = data[data['cluster'] == 3]

    # plot histograms
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data=c0_df['light_sleep_breathing_rate'], label='C0', color='#9999ff', fill=True, alpha=0.6)
    sns.kdeplot(data=c1_df['light_sleep_breathing_rate'], label='C1', color='#ffff66', fill=True, alpha=0.6)
    sns.kdeplot(data=c2_df['light_sleep_breathing_rate'], label='C2', color='#66ffb5', fill=True, alpha=0.6)
    sns.kdeplot(data=c3_df['light_sleep_breathing_rate'], label='C3', color='#ffb366', fill=True, alpha=0.6)
    plt.xlabel('light sleep breathing rate')
    plt.ylabel('frequency')
    plt.legend(frameon=False)
    plt.show()
