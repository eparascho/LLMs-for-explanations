import pandas as pd


''' 
This function generates a prompt in the zero-shot learning setting for a given instance in the dataset.
'''
def zero_prompt(data, instance, target, target_encoding, granularity, task):
    # standard prompt part
    zero_prompt = ("Why a user on " + pd.to_datetime(data.loc[instance, 'date']).day_name()) 

    # granularity prompt part
    if granularity == 'hourly':
        time_part = (" at " + str(pd.to_datetime(data.loc[instance, 'date']).hour) + " o'clock")
        zero_prompt += time_part

    # starting the features prompt part
    features_intro_part = (", who has ") 
    zero_prompt += features_intro_part

    # features prompt part
    features = list(data.columns)
    for feature in features:
        if feature != 'id' and feature != 'date' and feature != target:
            if feature == features[-2]:
                feature_part = ("and " + str(data.loc[instance, feature]) + " " + feature + ", has ")
                zero_prompt += feature_part
            else:
                feature_part = (str(data.loc[instance, feature]) + " " + feature + ", ")
                zero_prompt += feature_part

    # target prompt part
    if data.loc[instance, target] == list(target_encoding.keys())[0]:
        target_part = (str(target_encoding[0]) + " ")
    else:
        target_part = (str(target_encoding[1]) + " ")
    zero_prompt += target_part

    # task prompt part
    zero_prompt += (str(task) + "?")

    # replace all '_' with ' ' for better readability
    zero_prompt = zero_prompt.replace('_', ' ')

    return zero_prompt