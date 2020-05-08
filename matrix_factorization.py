import numpy as np
import pandas as pd
import pymc3 as pm
import logging
import theano
import warnings
import evalution_metrics as em

warnings.filterwarnings("ignore")

# Enable on-the-fly graph computations, but ignore
# absence of intermediate test values.
theano.config.compute_test_value = 'ignore'
# Set up logging.
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class PMF:
    """Probabilistic Matrix Factorization Model."""

    def __init__(self, R, dim, alpha=2, std=0.01, bounds=(1, 5)):
        """
        :param np.ndarray R: The training data to use for learning of the model .
        :param int dim: latent dimensionality of the model
        :param int alpha: Fixed precision for the likelihood function.
        :param float std: Amount of noise to use for model initialization.
        :param (tuple of int) bounds: (lower, upper) bound of ratings used for capping the estimates produced for rating matrix.
        """
        self.dim = dim
        self.alpha = alpha
        self.std = np.sqrt(1.0 / alpha)
        self.bounds = bounds
        self.data = R.copy()
        n, m = self.data.shape

        # Performing mean value imputation
        # Test element-wise whether it is NaN or not return the result as a boolean array
        nan_mask = np.isnan(self.data)
        # Filling the NaN values
        self.data[nan_mask] = self.data[~nan_mask].mean()

        # Low precision reflects uncertainty; prevents overfitting.
        # Setting the precision to the mean variance across users and items.
        self.alpha_u = 1 / self.data.var(axis=1).mean()
        self.alpha_v = 1 / self.data.var(axis=0).mean()

        # Number of rows(users) and columns(items) of the rating matrix
        num_rows = len(R)
        num_cols = len(R[0])
        U = np.random.rand(num_rows, dim)
        V = np.random.rand(num_cols, dim)

        # Building the model.
        logging.info('Building the Probabilistic Matrix Factorization model')
        with pm.Model() as pmf:
            # Each row of U is drawn from a multivariate normal distribution with mean μ=0 (mu=0).
            # Precisions αU for U and αV for V are some multiples of the identity matrix I(alpha_u, alpha_v)
            U = pm.MvNormal(
                'U', mu=0, tau=self.alpha_u * np.eye(dim),
                shape=(n, dim), testval=np.random.randn(n, dim) * std)
            V = pm.MvNormal(
                'V', mu=0, tau=self.alpha_v * np.eye(dim),
                shape=(m, dim), testval=np.random.randn(m, dim) * std)
            R = pm.Normal(
                'R', mu=(U @ V.T)[~nan_mask], tau=self.alpha,
                observed=self.data[~nan_mask])

        logging.info('Completed building the Probabilistic Matrix Factorization model')
        self.model = pmf

    def __str__(self):
        return self.name

    def _map(self):
        try:
            return self._map
        except:
            return self.find_map()
    map = property(_map)

    def find_map(self):
        """Find Maximum A Posterior (MAP) using L-BFGS-B optimization."""
        with self.model:
            logging.info('finding PMF MAP using L-BFGS-B optimization...')
            self._map = pm.find_MAP(method='L-BFGS-B')

        logging.info('Found PMF MAP successfully.')
        return self._map

    def predict(self, U, V):
        """Predict the Rating matrix for the given User and Item matrices."""
        R = np.dot(U, V.T)
        n, m = R.shape
        sample_R = np.random.normal(R, self.std)
        # bound ratings
        low, high = self.bounds
        sample_R[sample_R < low] = low
        sample_R[sample_R > high] = high
        return sample_R

    def eval_map(pmf_model, train, test):
        """Evaluate the MAP estimates by computing RMSE on the predicted ratings obtained from the MAP values of U
        and V."""
        U = pmf_model.map['U']
        V = pmf_model.map['V']

        # Making predictions on the U and V given by MAP
        predictions = pmf_model.predict(U, V)
        # Calculating RMSE and MAE on predictions and the train, test datasets.
        train_rmse = em.rmse(train, predictions)
        test_rmse = em.rmse(test, predictions)
        train_mae = em.mae(train, predictions)
        test_mae = em.mae(test, predictions)
        overfit_rmse = test_rmse - train_rmse
        overfit_mae = test_mae - train_mae

        print('RMSE for Probabilistic Matrix Factorization MAP training: %.5f' % train_rmse)
        print('RMSE for Probabilistic Matrix Factorization MAP testing:  %.5f' % test_rmse)
        print('MAE for Probabilistic Matrix Factorization MAP training: %.5f' % train_mae)
        print('MAE for Probabilistic Matrix Factorization MAP testing:  %.5f' % test_mae)
        print('Overfitting found with RMSE (Train/test difference): %.5f' % overfit_rmse)
        print('Overfitting found with MAE (Train/test difference): %.5f' % overfit_mae)

        return test_rmse, test_mae

    def draw_samples(self, **kwargs):
        """Draw MCMC samples. the sampling infrastructure."""
        kwargs.setdefault('chains', 1)
        with self.model:
            self.trace = pm.sample(**kwargs)

    def norms(pmf_model, monitor=('U', 'V'), ord='fro'):
        """Return norms of latent variables at each step in the sample trace. These can be used to monitor
        convergence of the sampler. """
        monitor = ('U', 'V')
        norms = {var: [] for var in monitor}
        for sample in pmf_model.trace:
            for var in monitor:
                norms[var].append(np.linalg.norm(sample[var], ord))
        return norms

    def running_rmse(pmf_model, test_data, train_data, burn_in=0, plot=True):
        """Calculate RMSE for each step of the trace to monitor convergence."""
        burn_in = burn_in if len(pmf_model.trace) >= burn_in else 0
        results = {'per-step-train': [], 'running-train': [],
                   'per-step-test': [], 'running-test': []}
        R = np.zeros(test_data.shape)
        for cnt, sample in enumerate(pmf_model.trace[burn_in:]):
            sample_R = pmf_model.predict(sample['U'], sample['V'])
            R += sample_R
            running_R = R / (cnt + 1)
            results['per-step-train'].append(em.rmse(train_data, sample_R))
            results['running-train'].append(em.rmse(train_data, running_R))
            results['per-step-test'].append(em.rmse(test_data, sample_R))
            results['running-test'].append(em.rmse(test_data, running_R))

        results = pd.DataFrame(results)

        if plot:
            results.plot(
                kind='line', grid=False, figsize=(15, 7),
                title='Per-step and Running RMSE From Posterior Predictive')

        # Return the final predictions, and the RMSE calculations
        return running_R, results

