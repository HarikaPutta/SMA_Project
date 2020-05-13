import numpy as np


def RMSE(actual, prediction):
    """Calculate Root Mean Square Error"""
    I = ~np.isnan(actual)  # indicator for missing values
    sqerror = abs(actual - prediction) ** 2  # squared error array
    mse = np.nanmean(sqerror[I])  # mean squared error
    return np.sqrt(mse)


def MAE(actual, prediction):
    """Calculate Mean Absolute Error."""
    I = ~np.isnan(actual)
    error = abs(actual - prediction)  # squared error array
    mae = np.nanmean(error[I])  # mean error
    return mae
