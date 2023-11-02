"""
This function ...
"""


def inverse_score(score, min, max):
    return max - score + min


"""
This function ...
"""


def personality_category(score, mean, std):
    if score > mean + 0.5 * std:
        return 'HIGH'
    if score < mean - 0.5 * std:
        return 'LOW'
    return 'AVERAGE'


"""
This function ...
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
This function ...
"""


def convert_5_to_4_likert(x):
    return (4 - 1) * (x - 1) / (5 - 1) + 1


"""
This function ...
"""
def proper_round(num, dec=0):
    num = str(num)[:str(num).index('.') + dec + 2]
    if num[-1] >= '5':
        a = num[:-2 - (not dec)]  # integer part
        b = int(num[-2 - (not dec)]) + 1  # decimal part
        return float(a) + b ** (-dec + 1) if a and b == 10 else float(a + str(b))
    return float(num[:-1])

"""
This function ...
"""
def get_stai_category(score, mean_stai, std_stai):
    if score < mean_stai-0.5*std_stai:
        return "Below average"
    if score > mean_stai+0.5*std_stai:
        return "Above average"
    return "Average"
