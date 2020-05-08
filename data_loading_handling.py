import pandas as pd
import numpy as np


# Load the dataset
def load_dataset():
    headers = ['userId', 'movieId', 'genreID', 'reviewId', 'movieRating', 'reviewDate']
    columns = ['userId', 'movieId', 'genreID', 'movieRating']
    data = pd.read_csv('Ciao-DVD-Datasets/movie-ratings.txt', sep=',', names=headers, usecols=columns,
                       dtype={"userId": "str", "movieId": "str"})

    # Getting the basic information about the data
    num_users = data.userId.unique().shape[0]
    num_items = data.movieId.unique().shape[0]
    sparsity = 1 - len(data) / (num_users * num_items)
    print(f"Users: {num_users}\nMovies: {num_items}\nSparsity: {sparsity}")

    # Returning the dataset
    return data


# Select some random data from the entire dataset
def select_random_data(data, percent_test):
    selected_data = data.sample(frac=percent_test)
    return selected_data


# Split the data into train(80%) and test(20%)
def split_train_test_custom(data, percent_test):
    n, m = data.shape  # # users, # movies
    N = n * m  # # cells in matrix

    # Preparing train/test ndarrays.
    train = data.copy()
    test = np.ones(data.shape) * np.nan

    # Drawing random sample of training data to use for testing.
    tosample = np.where(~np.isnan(train))  # ignore nan values in data
    idx_pairs = list(zip(tosample[0], tosample[1]))  # tuples of row/col index pairs

    test_size = int(len(idx_pairs) * percent_test)  # use 20% of data as test set
    train_size = len(idx_pairs) - test_size  # and remainder for training

    indices = np.arange(len(idx_pairs))  # indices of index pairs
    sample = np.random.choice(indices, replace=False, size=test_size)

    # Transferring random sample from train set to test set.
    for idx in sample:
        idx_pair = idx_pairs[idx]
        test[idx_pair] = train[idx_pair]  # transfer to test set
        train[idx_pair] = np.nan  # remove from train set

    # Verifying everything worked properly
    assert (train_size == N - np.isnan(train).sum())
    assert (test_size == N - np.isnan(test).sum())

    # Returning train set and test set
    return train, test


def data_analysis(data):
    # Extract the ratings from the DataFrame
    ratings = data.movieRating
    # Plot histogram
    data.groupby('rating').size().plot(kind='bar');
    data.movieRating.describe()
    movie_means = data['movieId'].groupby('movieId').movieRating.mean()
    movie_means[:50].plot(kind='bar', grid=False, figsize=(16, 6),
                          title="Mean ratings for 50 movies");
