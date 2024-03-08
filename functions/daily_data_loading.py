import pandas as pd


"""
This function processes the datetime object by converting it to timestamp object and extracting the day.
"""
def date_conversion(df):
    df["date"] = pd.to_datetime(pd.to_datetime(df["date"]).dt.date)

    return df
