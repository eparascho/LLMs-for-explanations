"""
This file contains the code for visualizing the categorical features before converting into numerical for finding the statistical difference.
"""

import warnings
import pandas as pd
from functions.visualizations import bar_plot
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    data = pd.read_pickle('../../data/labeling_visualizations/kmeans_clean_labeling.pkl')

    # categorical features
    categorical_features = ['ecg', 'heart_rate_alert', 'gender', 'age', 'bmi', 'extraversion', 'agreeableness',
                            'conscientiousness', 'stability', 'intellect', 'self_determination',
                            'positive_affect_score', 'negative_affect_score', 'stai_stress', 'ttm_stage',
                            'dramatic_relief_category', 'environmental_reevaluation_category',
                            'self_reevaluation_category',
                            'social_liberation_category', 'reinforcement_management_category',
                            'self_liberation_category', 'mood']

    for categorical in categorical_features:
        # keep only the data that does not contain NaN values in this specific feature
        visualize_data = data.dropna(subset=[categorical])
        visualize_data['cluster'] = visualize_data['cluster'].astype(str)
        bar_plot(visualize_data, categorical, 'kmeans', 'clean')
