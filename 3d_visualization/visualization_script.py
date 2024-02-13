"""
This file visualizes the relationship of training features with the labeling features providing a 3d visualization
"""


import numpy as np
import pandas as pd
import plotly.express as px


if __name__ == '__main__':
    # read data
    df = pd.read_pickle('../data/labeling_visualizations/kmeans_clean_labeling.pkl')
    print(df.columns)

    # it is supported only for numerical features
    labeling_features = ['nightly_temperature', 'nremhr', 'spo2', 'rmssd', 'full_sleep_breathing_rate', 'deep_sleep_breathing_rate',
       'light_sleep_breathing_rate', 'rem_sleep_breathing_rate', 'stress_score', 'responsiveness_points', 'wrist_temperature', 'vo2max', 'oxygen_variation',
       'scl_avg', 'mindfulness_start_heart_rate', 'mindfulness_end_heart_rate', 'resting_heart_rate', 'mood_value', 'bpm']

    # add jitter factor to the data
    visualize_df = df.copy()
    jitter_factor = 1.2
    for column in labeling_features:
        visualize_df[column] = visualize_df[column] + np.random.uniform(-jitter_factor, jitter_factor, size=visualize_df[column].shape)

    # run the 3d plot
    visualize_df.dropna(subset=['stress_score'], inplace=True)
    fig = px.scatter_3d(visualize_df, x='steps', y='calories', z='very_active_minutes', color='stress_score', color_continuous_scale=px.colors.sequential.Viridis, opacity=0.7)
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(scene=dict(xaxis_title='Steps', yaxis_title='Calories', zaxis_title='Very Active Minutes'))
    fig.show()
