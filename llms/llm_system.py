
def create_system(ml_task, granularity, target_encoding, target, real_task, learning, scope, xai_method, examples):
    # ml task system content
    system_content = ("You are a XAI model that can help me explain the " + ml_task + " results of my data. ")

    # granularity system content
    system_content += ("I have a dataset with " + granularity + " wearable data. ")
                      
    # target system content
    system_content += ("The " + ml_task + " algorithm categorized the data into " + str(len(target_encoding)) + " " +  target + "s, where " + target + " " +  str(list(target_encoding.keys())[0]) + " represents " + \
    target_encoding[0] + " " + real_task + " and " + target + " " +  str(list(target_encoding.keys())[1]) + " represents " + target_encoding[1] + " " + real_task + ". ")
    
    # xai task system content
    system_content += ("I need to understand why a user over time has been categorized into its respective " + target + ". You will be provided with ")

    # learning technique system content
    if learning == 'one':
        system_content += examples + " example which "
    elif learning == 'few':
        system_content += examples + " examples each one "

    if learning == 'one' or learning == 'few':
        system_content += ("contains, in the first sentence, the features, values and " + ml_task + " result, and, in a second sentence, the explanation produced by the " + scope + " " + xai_method + " " + \
        "XAI method, which is based on feature importance, to explain this " + ml_task + " result. After the ")

    if learning == 'one':
        system_content += "example, there will be "
    elif learning == 'few':
        system_content += "examples, there will be "
     
    # standard technique system content
    system_content += ("a question containing features and their actual values. You need to compute the feature importance and explain the " + ml_task + " results based on this feature importance. " + \
    "Your answer must contain only the exact following two parts: " + \
    "The \"Developer response:\" as exclusively a json format with keys to be the features you identified in the question and values to be their feature importance you computed. For example: \"steps\":0.1. Do not add any other information. " + \
    "The \"User response:\" in one short paragraph in which you explain the " + ml_task + " results based on the feature importance you computed. The answer must include the features you identified along with their actual values. " +\
    "Answer in a consistent style, with clear, short and understandable sentences.")

    return system_content