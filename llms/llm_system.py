
def create_system(ml_task, granularity, target_encoding, target, real_task):
    # ml task system content
    system_content = ("You are a XAI model that can help me explain the " + ml_task + " results of my data. ")

    # granularity system content
    system_content += ("I have a dataset with " + granularity + " wearable data. ")
                      
    # target system content
    system_content += ("The " + ml_task + " algorithm categorized the data into " + str(len(target_encoding)) + " " +  target + "s, where " + target + " " +  str(list(target_encoding.keys())[0]) + " represents " + \
    target_encoding[0] + " " + real_task + " and " + target + " " +  str(list(target_encoding.keys())[1]) + " represents " + target_encoding[1] + " " + real_task + ". ")
    
    # task system content
    system_content += ("I need to understand why a user over time has been categorized into its respective " + target + ". You will be provided with a text containing features and their actual values. " + \
    "You need to compute the feature importance and explain the " + ml_task + " results based on this feature importance. Your answer must contain only the exact following two parts: " + \
    "The \"Developer response:\" as exclusively a json format with keys to be the features you identified in the text and values to be their feature importance you computed. For example: \"steps\":0.1. Do not add any other information. " + \
    "The \"User response:\" in one short paragraph in which you explain the " + ml_task + " results based on the feature importance you computed. The answer must include the features you identified along with their actual values. " +\
    'Answer in a consistent style.')

    return system_content