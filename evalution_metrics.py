import numpy as np
import pandas as pd


class MeanOfMeansBaseline:
    """Mean of Means baseline method used for comparision of the results of PMF to know its goodness.
    Filling the missing values using mean of user/item/global means."""

    def __init__(self, train_data):
        """Mean of Means baseline method used for comparision of the results of PMF to know its goodness."""
        self.predict(train_data.copy())

    def predict(self, train_data):
        """Filling the missing values using mean of user/item/global means"""
        nan_mask = np.isnan(train_data)
        masked_train = np.ma.masked_array(train_data, nan_mask)
        global_mean = masked_train.mean()
        user_means = masked_train.mean(axis=1)
        item_means = masked_train.mean(axis=0)
        self.predicted = train_data.copy()
        n, m = train_data.shape
        for i in range(n):
            for j in range(m):
                if np.ma.isMA(item_means[j]):
                    self.predicted[i, j] = np.mean(
                        (global_mean, user_means[i]))
                else:
                    self.predicted[i, j] = np.mean(
                        (global_mean, user_means[i], item_means[j]))

    def rmse(self, test_data):
        """Calculating the RMSE for predictions on test data."""
        return rmse(test_data, self.predicted)

    def mae(self, test_data):
        """Calculating the MAE for predictions on test data."""
        return mae(test_data, self.predicted)

    def __str__(self):
        return 'Baseline'


def rmse(test_data, predicted):
    """Calculate Root Mean Square Error by ignoring missing values in the test data."""
    I = ~np.isnan(test_data)  # indicator for missing values
    N = I.sum()  # number of non-missing values
    sqerror = abs(test_data - predicted) ** 2  # squared error array
    mse = sqerror[I].sum() / N  # mean squared error
    return np.sqrt(mse)


def mae(test_data, predicted):
    """Calculate Mean Absolute Error by ignoring missing values in the test data."""
    I = ~np.isnan(test_data)
    N = I.sum()  # number of non-missing values
    error = abs(test_data - predicted)  # squared error array
    mae = (error[I].sum() / N).sum()
    return mae
