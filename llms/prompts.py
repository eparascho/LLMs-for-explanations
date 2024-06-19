import json
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


'''
This function generates a prompt for an example fro the one-shot and few-shot learning setting respectively.
'''
def example_prompt(data, example_instance, target, target_encoding, granularity, task, xai_response):
    # standard prompt part
    example_prompt = ("A user on " + pd.to_datetime(data.loc[example_instance, 'date']).day_name())

    # granularity prompt part
    if granularity == 'hourly':
        time_part = (" at " + str(pd.to_datetime(data.loc[example_instance, 'date']).hour) + " o'clock")
        example_prompt += time_part

    # starting the features prompt part
    features_intro_part = (", who has ") 
    example_prompt += features_intro_part

    # features prompt part
    features = list(data.columns)
    for feature in features:
        if feature != 'id' and feature != 'date' and feature != target:
            if feature == features[-2]:
                feature_part = ("and " + str(data.loc[example_instance, feature]) + " " + feature + ", has also ")
                example_prompt += feature_part
            else:
                feature_part = (str(data.loc[example_instance, feature]) + " " + feature + ", ")
                example_prompt += feature_part

    # target prompt part
    if data.loc[example_instance, target] == list(target_encoding.keys())[0]:
        target_part = (str(target_encoding[0]) + " ")
    else:
        target_part = (str(target_encoding[1]) + " ")
    example_prompt += target_part
    
    # task prompt part
    example_prompt += (str(task) + ". ")

    # explanation prompt part
    example_prompt += ("The explanation for this user's " + task + " gives the following feature importances: " + str(xai_response) + " . ") 

    return example_prompt


''' 
This function generates a prompt in the one-shot learning setting for a given instance in the dataset.
'''
def one_prompt(data, instance_interpret, example_instance, target, target_encoding, granularity, task):    
    # the one example prompt part
    with open(f'../data/explainability_output/local_lime_{example_instance}.json') as f:
        xai_response = json.load(f)
    one_prompt = example_prompt(data, example_instance, target, target_encoding, granularity, task, xai_response)

    # the zero-shot learning prompt part
    one_prompt += zero_prompt(data, instance_interpret, target, target_encoding, granularity, task)

    return one_prompt


''' 
This function generates a prompt in the few-shot learning setting for a given instance in the dataset.
'''
def few_prompt(data, instance_interpret, example_instances, target, target_encoding, granularity, task):
    # the few example prompt part
    few_prompt = ""
    for example_instance in example_instances:
        with open(f'../data/explainability_output/local_lime_{example_instance}.json') as f:
            xai_response = json.load(f)   
        few_prompt += example_prompt(data, example_instance, target, target_encoding, granularity, task, xai_response)

    # the zero-shot learning prompt part
    few_prompt += zero_prompt(data, instance_interpret, target, target_encoding, granularity, task)

    return few_prompt