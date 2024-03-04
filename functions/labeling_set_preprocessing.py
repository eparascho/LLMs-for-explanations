"""
This file contains the function that converts the categorical features to numerical based on their nature (ordinal, binary, etc.)
"""


def convert_categorical(data):
    # ordinal features
    ordinals = ['positive_affect_score', 'negative_affect_score', 'stai_stress', 'extraversion', 'agreeableness', 'conscientiousness', 'stability', 'intellect', 'dramatic_relief_category',
               'environmental_reevaluation_category', 'self_reevaluation_category', 'social_liberation_category', 'reinforcement_management_category', 'self_liberation_category']
    order_mapping = {'Below average': 1, 'Average': 2, 'Above average': 3}
    for ordinal in ordinals:
        data[ordinal] = data[ordinal].map(order_mapping)

    # for the ttm_stage feature
    order_mapping = {'Precontemplation': 1, 'Contemplation': 2, 'Preparation': 3, 'Action': 4, 'Maintenance': 5}
    data['ttm_stage'] = data['ttm_stage'].map(order_mapping)

    # for the self_determination feature
    order_mapping = {'amotivation': 1, 'external_regulation': 2, 'introjected_regulation': 3, 'identified_regulation': 4, 'intrinsic_regulation': 5}
    data['self_determination'] = data['self_determination'].map(order_mapping)

    # for the bmi feature
    order_mapping = {'Underweight': 1, 'Normal': 2, 'Overweight': 3, 'Obese': 4}
    data['bmi'] = data['bmi'].map(order_mapping)

    # binary features
    binaries = ['ecg', 'heart_rate_alert', 'gender', 'age']
    for binary in binaries:
        category_order = data[binary].value_counts().index
        mapping = {category: index for index, category in enumerate(category_order)}
        data[binary] = data[binary].map(mapping)

    # for the mood feature
    order_mapping = {'<no-response': 0, 'FEAR': 1, 'ANGER': 2, 'TENSE/ANXIOUS': 3, 'SAD': 4, 'TIRED': 5, 'SURPRISE': 6, 'ALERT': 7, 'NEUTRAL': 8, 'JOY': 9, 'HAPPY': 10, 'RESTED/RELAXED': 11}
    data['mood'] = data['mood'].map(order_mapping)

    return data
