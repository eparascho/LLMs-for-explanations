import json


'''
This function stores the LLM response in a file.
'''
def store_response(profile_response, model, learning, instance, profile):
    if profile == 'user':
        response_path = '../data/llms_output/' + model + '_' + learning + '_' + instance + '_' + profile + '.txt'
        with open(response_path, 'w') as f:
            f.write(profile_response)
    else:
        response_path = '../data/llms_output/' + model + '_' + learning + '_' + instance + '_' + profile + '.json'
        with open(response_path, 'w') as f:
            json.dump(profile_response, f)


'''
This function processes the LLM response for the developer and prepares it for storing.
'''
def developer_response_processing(response):
    developer_response = response.split("User response:", 1)[0].strip()
    developer_response = developer_response.replace('\n', '')
    if '```json' in developer_response:
        developer_response = developer_response.split("```json", 1)[1]
        developer_response = '{' + ''.join(developer_response) + '}'
        developer_response = developer_response.replace('```', '')
    else:
        if 'Developer response:' in developer_response:
            developer_response = developer_response.split("Developer response:", 1)[1]
        elif 'Developer response: ' in developer_response:
            developer_response = developer_response.split("Developer response: ", 1)[1]
        elif ' Developer response:' in developer_response:
            developer_response = developer_response.split(" Developer response:", 1)[1]
        elif ' Developer response: ' in developer_response:
            developer_response = developer_response.split(" Developer response: ", 1)[1]
    developer_response = json.loads(developer_response)

    return developer_response