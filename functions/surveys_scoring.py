"""
This function inverses the scores of specific questions on each questionnaire from negative to positive. In more detail, there are answers
that contribute negatively to the final result so that they need to be converted before summing the individual scores of a questionnaire.
The function returns the inversed score.
"""
def inverse_score(score, min, max):
    return max - score + min


"""
This function identifies the category of a specific value. In more detail, each value corresponds to
a specific category/level of a distribution defined by the mean and std of all the values.
The function returns the score's category (above average, below average or average).
"""
def get_category(score, mean, std):
    if score > mean + 0.5 * std:
        return 'Above average'
    if score < mean - 0.5 * std:
        return 'Below average'
    return 'Average'


"""
This function encodes the detailed stage of changes of the ttm questionnaire into a stage-word.
The function returns the stage-word (Precontemplation, Contemplation, Preparation, Action or Maintenance).
"""
def define_stage_of_change(response):
    if response == "No, and I do not intend to do regular physical activity in the next 6 months.":
        return "Precontemplation"
    if response == "No, but I intend to do regular physical activity in the next 6 months.":
        return "Contemplation"
    if response == "No, but I intend to do regular physical activity in the next 30 days.":
        return "Preparation"
    if response == "Yes, I have been doing physical activity regularly, but for less than 6 months.":
        return "Action"
    return "Maintenance"


"""
This function converts the scale of the STAI answers from 5-likert scale to 4-likert scale.
The function returns the converted score.
"""
def convert_5_to_4_likert(x):
    return (4 - 1) * (x - 1) / (5 - 1) + 1


"""
This function implements the rounding of values that are decimal but they need to be integer.
The function returns the rounded score.
"""
def proper_round(num, dec=0):
    num = str(num)[:str(num).index('.') + dec + 2]
    if num[-1] >= '5':
        a = num[:-2 - (not dec)]  # integer part
        b = int(num[-2 - (not dec)]) + 1  # decimal part
        return float(a) + b ** (-dec + 1) if a and b == 10 else float(a + str(b))
    return float(num[:-1])
