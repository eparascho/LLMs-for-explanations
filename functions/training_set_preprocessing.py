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
    experiments_df = cleaned_training_df.loc[((cleaned_training_df['date'] > '2021-05-23') & (cleaned_training_df['date'] < '2021-07-27')) | (
                    (cleaned_training_df['date'] > '2021-11-14') & (cleaned_training_df['date'] < '2022-01-18'))]
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
