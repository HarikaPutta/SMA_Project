import data_loading_handling as dlh
from matrix_factorization import PMF
import evalution_metrics as em

import warnings
warnings.filterwarnings("ignore")

# Fixed precision for the likelihood function.
ALPHA = 2
# Amount of noise to use for model initialization.
STD = 0.05
# Latent dimensionality of the model
DIM = 10


# Loading the dataset
data = dlh.load_dataset()
# Selecting random rows from the entire dataset
selected_data = dlh.select_random_data(data, 0.1)

# Getting the reshaped DataFrame organized by given index / column values.
dense_data = selected_data.pivot_table(index='userId', columns='movieId', values='movieRating').values

# Getting the training and testing datasets
train, test = dlh.split_train_test_custom(dense_data, 0.2)

# Training the PMF model with the train dataset.
pmf = PMF(train, DIM, ALPHA, std=0.05)

# Find mode of posterior using optimization.
pmf.find_map()

# Evaluating the estimates given by MAP of PMF.
pmf_map_rmse, pmf_map_mae = pmf.eval_map(train, test)

# Calculating the error using a baseline model to compare against the PMF
# Training the model over the train dataset
train_mom = em.MeanOfMeansBaseline(train)

# Calculating the RMSE error of the model using the test dataset
baseline_rmse = train_mom.rmse(test)
baseline_mae = train_mom.mae(test)
print('RMSE against baseline Mean Of Means method:\t%.5f' % baseline_rmse)
print('MAE against baseline Mean Of Means method:\t%.5f' % baseline_mae)

# Comparing the PMF against the baseline model
pmf_improvement_rmse = baseline_rmse - pmf_map_rmse
pmf_improvement_mae = baseline_mae - pmf_map_mae
print('PMF MAP improvement over the baseline model for RMSE:   %.5f' % pmf_improvement_rmse)
print('PMF MAP improvement over the baseline model for MAE:   %.5f' % pmf_improvement_mae)

# Draw MCMC samples.
pmf.draw_samples(draws=500, tune=500, )

predicted, results = pmf.running_rmse(test, train)

# Final RMSE results
final_test_rmse = results['running-test'].values[-1]
final_train_rmse = results['running-train'].values[-1]
print('Posterior predictive train RMSE: %.5f' % final_train_rmse)
print('Posterior predictive test RMSE:  %.5f' % final_test_rmse)
print('Train/test difference:           %.5f' % (final_test_rmse - final_train_rmse))
print('Improvement from MAP:            %.5f' % (pmf_map_rmse - final_test_rmse))
print('Improvement from Mean of Means:  %.5f' % (baseline_rmse - final_test_rmse))



