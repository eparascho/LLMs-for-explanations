import pandas as pd

"""
This function receives a list of column in the input and aggregates the values separately for each column/feature
If it is numerical it calculates the average, and if it is categorical the most frequent value.
"""
def aggregate_column(df, features):
    for el in ['id', 'date', 'hour']:
        features.remove(el)

    aggregated_df = pd.DataFrame(columns=['id', 'date', 'hour'])
    for feature in features:
        if isinstance(df[feature].iloc[0], int) or isinstance(df[feature].iloc[0], float):
            df_feature = df.groupby(['id', 'date', 'hour'])[feature].mean().to_frame()
            df_feature.reset_index(drop=False, inplace=True)
            aggregated_df = aggregated_df.merge(df_feature, how='outer', on=['id', 'date', 'hour'])
        elif type(df[feature].iloc[0] == str):
            df_feature = df.groupby(['id', 'date', 'hour'])[feature].agg(lambda x: x.value_counts().index[0]).to_frame()
            df_feature.reset_index(drop=False, inplace=True)
            aggregated_df = aggregated_df.merge(df_feature, how='outer', on=['id', 'date', 'hour'])

    return aggregated_df


"""
This function processes the datetime object by converting it to timestamp object and extracting the day and the hour.
"""
def date_conversion(df):
    df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True, format='mixed')
    df["hour"] = df["date"].dt.hour
    df["date"] = pd.to_datetime(df["date"].dt.date, infer_datetime_format=True)

    return df
