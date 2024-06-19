import json
import pandas as pd


'''
This function loads and denormalizes the data for the LLM.
'''
def load_data():
    data = pd.read_pickle('../data/explainability_input/test_data.pkl')  # the test data as defined in the explanability component
    data_features = (data.columns).drop(['cluster'])  # final features
    data['id'] = data['id'].astype(str)
    data['date'] = pd.to_datetime(data['date'])
    # load the latest version of denormalized data
    old_data = pd.read_pickle('../data/preprocessing_temps/date_engineered_training_df.pkl')
    old_data['id'] = old_data['id'].astype(str)
    old_data['date'] = pd.to_datetime(old_data['date'])
    # keep only the final columns
    normalized_data = old_data[data_features]
    # keep only the final clustering rows
    normalized_data = pd.merge(normalized_data, data[['id', 'date', 'cluster']], on=['id', 'date'], how='inner')
    normalized_data[normalized_data.columns[2:-1]] = normalized_data[normalized_data.columns[2:-1]].apply(lambda x: round(x, 3))

    return normalized_data


'''
This function stores the LLM response in a file.
'''
def store_response(profile_response, model, learning, instance, profile):
    if profile == 'user':
        response_path = '../data/llms_output/' + model + '_' + learning + '/' + instance + '_' + profile + '.txt'
        with open(response_path, 'w') as f:
            f.write(profile_response)
    else:
        response_path = '../data/llms_output/' + model + '_' + learning + '/' + instance + '_' + profile + '.json'
        with open(response_path, 'w') as f:
            json.dump(profile_response, f)


'''
This function processes the LLM response for the developer and prepares it for storing.
'''
def developer_response_processing(response):
    developer_response = response.split("User response:", 1)[0].strip()
    developer_response = developer_response.replace('\n', '')
    developer_response = developer_response.replace('\\', '')
    developer_response = developer_response.replace('"{', '{')
    developer_response = developer_response.replace('}"', '}')

    if '```json' in developer_response:
        developer_response = developer_response.split("```json", 1)[1]
        developer_response = developer_response.replace('}```', '')
        developer_response = '{' + ''.join(developer_response) + '}'
    else:
        if 'Developer response:' in developer_response:
            developer_response = developer_response.split("Developer response:", 1)[1]
        elif 'Developer response: ' in developer_response:
            developer_response = developer_response.split("Developer response: ", 1)[1]
        elif ' Developer response:' in developer_response:
            developer_response = developer_response.split(" Developer response:", 1)[1]
        elif ' Developer response: ' in developer_response:
            developer_response = developer_response.split(" Developer response: ", 1)[1]
    
    if '{' not in developer_response:
        developer_response = '{' + ''.join(developer_response)
    elif '}' not in developer_response:
        if developer_response[-1] == '.':
            developer_response = ''.join(developer_response)[:-1] + '}'
        else:
            developer_response = ''.join(developer_response) + '}'

    developer_response = json.loads(developer_response)

    return developer_response


'''
This function processes the LLM response for the user and prepares it for storing.
'''
def postprocessing(response, model, learning, instance_interpret):
    
    if 'User response:' not in response:
        response = response.split('\n')
        response[-1] = 'User response: ' + response[-1]
        response = '\n'.join(response)

    profile = 'user'
    user_response = response.split("User response:", 1)[1].strip()
    store_response(user_response, model, learning, str(instance_interpret), profile)

    profile = 'developer'
    developer_response = developer_response_processing(response)
    store_response(developer_response, model, learning, str(instance_interpret), profile)

    return user_response, developer_response