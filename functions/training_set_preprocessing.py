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
