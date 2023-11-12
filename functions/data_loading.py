import pandas as pd

"""
This function receives a list of column in the input and aggregates the values separately for each column/feature
If all input features are numerical it calculates the average, and if all input features are categorical or 
some of the are categorical and other numerical it keeps all the values.
"""
def aggregate_column(df, features, state):
    for el in ['id', 'date', 'hour']:
        features.remove(el)

    aggregated_df = pd.DataFrame(columns=['id', 'date', 'hour'])

    if state == 'categoricals':
        for feature in features:
            if type(df[feature].iloc[0] == str):
                print("in categoricals")
                # get all possible values
                df_feature = df.groupby(['id', 'date', 'hour'])[feature].agg(lambda l: list(set(l)) if isinstance(l, list) else l).to_frame()
                df_feature.reset_index(drop=False, inplace=True)
                aggregated_df = aggregated_df.merge(df_feature, how='outer', on=['id', 'date', 'hour'])
            else:
                print("Wrong type provided!")

    elif state == 'numericals':
        for feature in features:
            if isinstance(df[feature].iloc[0], int) or isinstance(df[feature].iloc[0], float):
                print("in numericals")
                if feature in ['calories', 'distance', 'steps']:
                    # get the sum value for each hour
                    df_feature = df.groupby(['id', 'date', 'hour'])[feature].sum().to_frame()
                elif feature in ['altitude']:
                    # get the max value for each hour
                    df_feature = df.groupby(['id', 'date', 'hour'])[feature].max().to_frame()
                else:
                    # get the average value for each hour
                    df_feature = df.groupby(['id', 'date', 'hour'])[feature].mean().to_frame()
                df_feature.reset_index(drop=False, inplace=True)
                aggregated_df = aggregated_df.merge(df_feature, how='outer', on=['id', 'date', 'hour'])
            else:
                print("Wrong type provided!")

    elif state == 'categorical-numerical':
        for feature in features:
            if isinstance(df[feature].iloc[0], int) or isinstance(df[feature].iloc[0], float):
                print("in numericals part")
                # get the corresponding to categorical value
                df_feature = df.groupby(['id', 'date', 'hour'])[feature].agg(lambda l: list(set(l)) if isinstance(l, list) else l).to_frame()
                df_feature.reset_index(drop=False, inplace=True)
                aggregated_df = aggregated_df.merge(df_feature, how='outer', on=['id', 'date', 'hour'])
            elif type(df[feature].iloc[0] == str):
                print("in categoricals part")
                # get all possible values
                df_feature = df.groupby(['id', 'date', 'hour'])[feature].agg(lambda l: list(set(l)) if isinstance(l, list) else l).to_frame()
                df_feature.reset_index(drop=False, inplace=True)
                aggregated_df = aggregated_df.merge(df_feature, how='outer', on=['id', 'date', 'hour'])
            else:
                print("Wrong type provided!")

    elif state == 'sleep':
        for feature in features:
            if isinstance(df[feature].iloc[0], int) or isinstance(df[feature].iloc[0], float):
                print("in numericals part")
                if feature in ['deep', 'rem', 'light', 'wake']:
                    # get the sum value for each hour
                    df_feature = df.groupby(['id', 'date', 'hour'])[feature].sum().to_frame()
                else:
                    # get the first value for each hour
                    df_feature = df.groupby(['id', 'date', 'hour'])[feature].first().to_frame()
                df_feature.reset_index(drop=False, inplace=True)
                aggregated_df = aggregated_df.merge(df_feature, how='outer', on=['id', 'date', 'hour'])
            elif type(df[feature].iloc[0] == bool):
                print("in booleans part")
                # get the maximum frequency
                df_feature = df.groupby(['id', 'date', 'hour'])[feature].agg(lambda x: x.value_counts().index[0]).to_frame()
                df_feature.reset_index(drop=False, inplace=True)
                aggregated_df = aggregated_df.merge(df_feature, how='outer', on=['id', 'date', 'hour'])
            else:
                print("Wrong type provided!")

    return aggregated_df


"""
This function processes the datetime object by converting it to timestamp object and extracting the day and the hour.
"""
def date_conversion(df):
    df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True, format='mixed')
    df["hour"] = df["date"].dt.hour
    df["date"] = pd.to_datetime(df["date"].dt.date, infer_datetime_format=True)

    return df
