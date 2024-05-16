import pandas as pd

def zero_prompt(data, instance):
    if data.loc[instance, 'cluster'] == 1:  # cluster 1 is positive
        cluster = 'positive'
    else:  # cluster 0 is negative
        cluster = 'negative'
    zero_prompt = (
        "Why a user on " + pd.to_datetime(data.loc[instance, 'date']).day_name() + " at " + str(pd.to_datetime(data.loc[instance, 'date']).hour) + " o'clock, who has " +
        str(data.loc[instance, 'exertion_points']) + " exertion points, " + str(data.loc[instance, 'step_goal']) + " step goal, " + str(data.loc[instance, 'minutes_below_zone_1']) + 
        " minutes below zone 1, " + str(data.loc[instance, 'minutes_in_zone_1']) + " minutes in zone 1, " + str(data.loc[instance, 'steps']) + " steps, " + 
        str(data.loc[instance, 'very_active_minutes']) + " very active minutes, " + str(data.loc[instance, 'minutes_in_zone_2']) + " minutes in zone 2, " + str(data.loc[instance, 'minutes_in_zone_3']) + " minutes in zone 3, " +
        str(data.loc[instance, 'altitude']) + " altitude, " + str(data.loc[instance, 'lightly_active_minutes']) + " lightly active minutes, " + str(data.loc[instance, 'moderately_active_minutes']) 
        + " moderately active minutes, " + str(data.loc[instance, 'sedentary_minutes']) + " sedentary minutes, " + str(data.loc[instance, 'exercises']) + " exercises, " + 
        str(data.loc[instance, 'exercise_duration']) + " exercise duration, and " + str(data.loc[instance, 'sleep_points']) + " sleep points, " + str(data.loc[instance, 'sleep_duration']) + 
        " sleep duration, " + str(data.loc[instance, 'calories']) + " calories has " + cluster + " well-being?")
    return zero_prompt

