"""
This file visualizes the relationship of training features with the labeling features providing a 3d visualization
"""


import numpy as np
import pandas as pd
import plotly.express as px


if __name__ == '__main__':
    # read data
    df = pd.read_pickle('../data/labeling_visualizations/kmeans_categories_labeling.pkl')
    print(df.columns)

    # add jitter factor to the data
    jitter_factor = 1.2
    df['cluster'] = df['cluster'] + np.random.uniform(-jitter_factor, jitter_factor, size=df['cluster'].shape)

    # run the 3d plot
    fig = px.scatter_3d(df, x='gender', y='stress_score', z='resting_heart_rate', color='cluster', color_continuous_scale=px.colors.sequential.Viridis, opacity=0.5)
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(scene=dict(xaxis_title='Gender', yaxis_title='Stress score', zaxis_title='Resting heart rate'))
    fig.show()
