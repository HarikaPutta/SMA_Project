import pandas as pd
import numpy as np


def load_dataset_pmf():
    """Load the dataset"""
    headers = ['userId', 'movieId', 'genreID', 'reviewId', 'movieRating', 'reviewDate']
    columns = ['userId', 'movieId', 'movieRating']
    data = pd.read_csv('Ciao-DVD-Datasets/movie-ratings.txt', sep=',', names=headers, usecols=columns,
                       dtype={'userId': 'int', 'movieId': 'int'})
    data = data - [1, 1, 0]
    # Getting the basic information about the data
    num_users = data.userId.unique().shape[0]
    num_items = data.movieId.unique().shape[0]
    density = len(data) / (num_users * num_items)
    sparsity = 1 - density
    print(
        f"No.of Users: {num_users}\nNo.of Movies: {num_items}\nData Density: {round(density, 5)}\nData Sparsity: "
        f"{round(sparsity, 5)}")
    # Returning the dataset
    return data


def train_validate_test_split_pmf(data):
    """Split Split the data into train(60%), validation(20%) and test(20%)"""
    train, validate, test = np.split(data.sample(frac=1), [int(.6 * len(data)), int(.8 * len(data))])
    # Returning the data as ndarrays
    return train.values, validate.values, test.values
