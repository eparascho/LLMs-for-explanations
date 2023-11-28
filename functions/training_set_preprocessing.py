import holidays
import numpy as np
import pandas as pd

"""
This function encodes the columns of the features that contain np.ndarrays as values into single value columns. 
In more detail, it receives as input 2 columns one categorical and one numerical and it creates one column for each unique value in the 
categorical feature. Then it assigns the numerical value for the corresponding categorical feature. If in a row, a categorical feature appears
twice, it sums the corresponding numerical values and assigns the total value.
It returns the encoded columns of the features.
"""
def encode_row(row, features):
    categorical = row[features[0]]
    result = {}

    for feature in row.index[1:]:
        value = row[feature]

        if isinstance(value, float):
            categorical = [categorical]
            value = [value]

        for c, v in zip(categorical, value):
            if not pd.isna(c) and not pd.isna(v):
                if c in result:
                    result[c] += v
                else:
                    result[c] = v

    return pd.Series(result)


"""
This function sums all the individual values inside a np.ndarray value of a dataframe cell.
It returns the total single value for the cell.
"""
def replace_with_sum(value):
    if isinstance(value, np.ndarray):
        return np.sum(value)
    else:
        return value


"""
This function averages all the individual values inside a np.ndarray value of a dataframe cell.
It returns the total single value for the cell.
"""
def replace_with_avg(value):
    if isinstance(value, np.ndarray):
        return np.mean(value)
    else:
        return value


"""
This function finds and lists the features that contain ndarrays and series values and thus they need encoding.
It returns the list of these features, while it also prints the total number of them.
"""
def features_to_encode(training_df):
    features = list(training_df.columns)
    del features[0:3]

    preprocessing_features = []
    for feature in features:
        if training_df[feature].apply(lambda d: True if isinstance(d, np.ndarray) or isinstance(d, pd.Series) else False).any():
            preprocessing_features.append(feature)
    print("The features that need encoding are:", len(preprocessing_features))
    return preprocessing_features


"""
This function calculates statistics and infers information about the experiments dates. In more detail, it finds how many records there are 
out of the official experiments dates, the users that provided information out of the experiment dates, as well as if any user participated 
both in the first and second round of the experiment. 
It prints several results, while it does not return something.
"""
def experiments_dates(cleaned_training_df):
    # prepare and sort the date column
    cleaned_training_df = cleaned_training_df.sort_values(by='date', ascending=True)
    cleaned_training_df['date'] = pd.to_datetime(cleaned_training_df['date'].astype("str"), format='%Y-%m-%d')

    # find the official experiment dates
    experiments_df = cleaned_training_df.loc[((cleaned_training_df['date'] > '2021-05-23') & (cleaned_training_df['date'] < '2021-07-27'))
                                             | ((cleaned_training_df['date'] > '2021-11-14') & (cleaned_training_df['date'] < '2022-01-18'))]
    experiments_df.reset_index(inplace=True, drop=True)

    # isolate the dates out of the official experiment
    extra = pd.concat([cleaned_training_df, experiments_df]).drop_duplicates(keep=False)
    print("There are", len(extra), "rows out of experiment dates from", extra['id'].nunique(), "unique users with 'normal distribution'.")

    # # find if a user was part of both first and second round - MAYBE SOMETHING WENT WRONG
    # first_round = cleaned_training_df.loc[(cleaned_training_df['date'] > '2021-05-23') & (cleaned_training_df['date'] < '2021-07-27')]
    # first_round.reset_index(inplace=True, drop=True)
    # first_users = set(list(first_round['id'].unique()))
    # second_round = cleaned_training_df.loc[(cleaned_training_df['date'] > '2021-11-14') & (cleaned_training_df['date'] < '2022-01-18')]
    # second_round.reset_index(inplace=True, drop=True)
    # second_users = set(list(second_round['id'].unique()))
    # if first_users & second_users:
    #     print("The user with id:", first_users & second_users, "provided data in both rounds of the experiment.")


"""
This function replaces all the NaN and 0 values of the given columns with the user's mean if it is not NaN, otherwise with all users mean.
It returns the dataframe filled with the corresponding mean values in these columns.
"""
def replace_nan_0_with_mean(training_df, features):
    for feature in features:
        # user's mean
        user_means = training_df.groupby('id')[feature].mean()
        # general mean for user's that do not have user mean
        user_means = user_means.fillna(training_df[feature].mean())
        # replace
        training_df[feature] = training_df.apply(lambda row: row[feature] if pd.notna(row[feature]) and row[feature] != 0
        else user_means[row['id']], axis=1)

    return training_df


"""
This function replaces all the NaN values of the given columns with 0 values.
It returns the dataframe filled with 0 values in these columns.
"""
def replace_nan_with_0(training_df, features):
    for feature in features:
        training_df[feature] = training_df[feature].fillna(0)

    return training_df


"""
This function replaces all the NaN values of the given columns with the user's mean if it is not NaN, otherwise with all users mean.
It returns the dataframe filled with the corresponding mean values in these columns.
"""
def replace_nan_with_mean(training_df, features):
    for feature in features:
        user_means = training_df.groupby('id')[feature].mean()
        training_df[feature] = training_df[feature].fillna(training_df['id'].map(user_means).fillna(training_df[feature].mean()))

    return training_df


"""
This function replaces all the NaN values of the given columns with the most common value.
It returns the dataframe filled with the corresponding common values in these columns.
"""
def replace_nan_with_common(training_df, features):
    for feature in features:
        training_df[feature] = training_df[feature].fillna(training_df[feature].mode().iloc[0])

    return training_df


"""
This function applies sin transformation to the hour or day to capture dates' cyclic behavior.
It returns the sin transformed hour or day value.
"""
def sin_transform(values):
    return np.sin(2 * np.pi * values / len(set(values)))


"""
This function applies cos transformation to the hour or day to capture dates' cyclic behavior.
It returns the cos transformed hour or day value.
"""
def cos_transform(values):
    return np.cos(2 * np.pi * values / len(set(values)))


"""
This function calculates if a day belongs to the weekend (1.0) dates or to the weekdays (0.0).
It returns the input dataframe with an additional column representing if it is weekend or weekdays.
"""
def is_weekend(df):
    df.loc[:, "is_weekend"] = df.date.dt.dayofweek  # returns 0-4 for Monday-Friday and 5-6 for Weekend
    df.loc[:, 'is_weekend'] = df['is_weekend'].apply(lambda d: 1.0 if d > 4 else 0.0)

    return df


"""
This function calculates if a day is holiday in Greece, Cyprus, Sweden or Italy, (1.0) or not (0.0).
It returns the input dataframe with an additional column representing if it is holiday or not.
"""
def is_holiday(df):
    gr_holidays = list(holidays.GR(years=[2021, 2022]).keys())
    swe_holidays = list(holidays.SWE(years=[2021, 2022]).keys())
    cy_holidays = list(holidays.CY(years=[2021, 2022]).keys())
    it_holidays = list(holidays.IT(years=[2021, 2022]).keys())

    df.loc[:, 'is_holiday'] = df.date.apply(lambda d: 1.0 if (
            (d in gr_holidays) or (d in swe_holidays) or (d in cy_holidays) or (d in it_holidays)) else 0.0)

    return df


"""
This function implements the date engineering of the dataframe. In more detail, it calculates the sin and cos transformations 
of the hour and day, it calculates if a day belongs to weekdays or to weekend, and if a day is holiday or not.
It returns the date engineered dataframe without the 'date' column and with the above mentioned ones.
"""
def date_engineering(df):

    # is weekend or weekday
    is_weekend(df)

    # is holiday or not
    is_holiday(df)

    # sin and cos transformations
    df["day"] = df["date"].apply(lambda x: x.day)
    df["hour"] = df["date"].apply(lambda x: x.hour)
    df["day_sin"] = sin_transform(df["day"])
    df["hour_sin"] = sin_transform(df["hour"])
    df["day_cos"] = cos_transform(df["day"])
    df["hour_cos"] = cos_transform(df["hour"])
    df = df.drop(columns=['date', 'day', 'hour'])

    return df
