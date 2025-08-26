import numpy as np
import warnings

from tqdm import tqdm

from ..log_likelihoods import Normal
from ..utils.mpi_utils import rank_zero_output_only
from scipy.spatial.distance import cdist, squareform, pdist
from sklearn.metrics.pairwise import pairwise_kernels


class VectorMCMC:
    def __init__(self, model, data, priors, log_like_args=None, log_like_func=Normal):
        """
        :param model: maps inputs to outputs
        :type model: callable
        :param data: data corresponding to model outputs
        :type data: 1D array
        :param priors: random variable objects with a pdf and rvs method (e.g.
            scipy stats random variable objects); note that the rvs method will
            also receive a numpy random number generator as an argument, which
            must be accounted for with custom prior objects
        :type priors: list of objects
        :param log_like_args: any fixed parameters that define the likelihood
            function (e.g., standard deviation for a Gaussian likelihood).
        :type log_like_args: 1D array or None
        :param log_like_func: log likelihood function that takes inputs, model,
            data, and hyperparameters and returns log likelihoods
        :type log_like_func: callable
        """
        self._eval_model = model
        self._data = data
        self._priors = priors
        self._log_like_func = log_like_func(self.evaluate_model, data, log_like_args)
        self._rng = np.random.default_rng()

    @property
    def rng(self):
        return self._rng

    @rng.setter
    def rng(self, rng):
        if isinstance(rng, np.random._generator.Generator):
            self._rng = rng
        else:
            raise TypeError("Random number generator must be a numpy generator.")

    def compute_covariance(self, inputs):
        def rbf(x, sigma):
            gamma = 1 / (2 * sigma**2)
            x = np.nan_to_num(x, nan=0.0, posinf=1e10, neginf=-1e10)
            return pairwise_kernels(x, metric="rbf", gamma=gamma)

        def median_dist(x):
            pairwise_dist = pdist(x, metric="euclidean")
            return np.median(pairwise_dist)

        D = inputs.shape[1]
        NU = 2.38 / np.sqrt(D)

        rbfs = rbf(inputs, sigma=median_dist(inputs))  # kernelernel grads

        R = NU**2 * np.array(
            [np.atleast_2d(np.cov(inputs.T, aweights=r)) for r in rbfs]
        )
        return R

    def smc_metropolis(self, inputs, num_samples, cov):
        num_particles = inputs.shape[0]
        log_priors, log_like = self._initialize_probabilities(inputs)
        cov = self.compute_covariance(inputs)
        scale = 1
        for i in range(num_samples):
            inputs, log_like, log_priors, rejected, newcov = self._perform_mcmc_step(
                inputs, cov, log_like, log_priors, scale
            )
            num_accepted = num_particles - np.sum(rejected)

            if num_accepted < inputs.shape[0] * 0.3:
                scale = 1 / 5
                cov = newcov * scale
            if num_accepted > inputs.shape[0] * 0.7:
                scale = 2
                cov = newcov * scale

        return inputs, log_like

    def metropolis(
        self,
        inputs,
        num_samples,
        cov,
        adapt_interval=None,
        adapt_delay=0,
        progress_bar=False,
        **kwargs,
    ):
        chain = np.zeros([inputs.shape[0], inputs.shape[1], num_samples + 1])
        chain[:, :, 0] = inputs

        log_priors, log_like = self._initialize_probabilities(inputs)

        for i in tqdm(range(1, num_samples + 1), disable=not progress_bar):
            inputs, log_like, log_priors, rejected, newcov = self._perform_mcmc_step(
                inputs, cov, log_like, log_priors
            )
            chain[:, :, i] = inputs

            cov = self.adapt_proposal_cov(cov, chain, i, adapt_interval, adapt_delay)
        return chain

    def evaluate_model(self, inputs):
        return self._eval_model(inputs)

    @rank_zero_output_only
    def sample_from_priors(self, num_samples):
        samples = []
        for i, p in enumerate(self._priors):
            samples.append(p.rvs(num_samples, random_state=self.rng).T)
        return np.vstack(samples).T

    def evaluate_log_priors(self, inputs):
        prior_dims = self._get_prior_dims()

        if inputs.shape[1] != sum(prior_dims):
            raise ValueError("Num prior distributions != num input params")

        log_priors = np.empty((inputs.shape[0], len(self._priors)))
        in_start_idx = 0
        for i, p in enumerate(self._priors):
            in_ = inputs[:, in_start_idx : in_start_idx + prior_dims[i]]
            log_priors[:, i] = p.logpdf(in_).squeeze()
            in_start_idx += prior_dims[i]

        return log_priors

    def _get_prior_dims(self):
        return [p.dim if hasattr(p, "dim") else 1 for p in self._priors]

    def evaluate_log_likelihood(self, inputs):
        log_like = self._log_like_func(inputs)
        return log_like.reshape(-1, 1)

    @staticmethod
    def evaluate_log_posterior(inputs, log_likelihood, log_priors):
        return np.sum(np.hstack((log_likelihood, log_priors)), axis=1)

    @rank_zero_output_only
    def proposal(self, inputs, cov):
        chol = np.array([self._ensure_psd_cov_and_do_chol_decomp(mat) for mat in cov])
        z = self.rng.normal(0, 1, inputs.shape)
        delta = np.einsum("ijk,ik->ij", chol, z)
        return inputs + delta

    def multivariate_normal_pdf_batch(self, data_points, means, covariances):
        """
        Computes the PDF of multiple N-dimensional multivariate normal distributions in batch.
        [Your provided function - keeping it unchanged]
        """
        M, N = data_points.shape

        dets = np.linalg.det(covariances)
        inv_covs = np.linalg.inv(covariances)
        norm_consts = 1.0 / np.sqrt((2 * np.pi) ** N * dets)

        deltas = data_points - means
        mahalanobis_dist_sq = np.einsum("mi,mij,mj->m", deltas, inv_covs, deltas)
        # mahalanobis_dist_sq = np.sum(deltas @ inv_covs * deltas, axis=1)

        pdf_vals = norm_consts * np.exp(-0.5 * mahalanobis_dist_sq)
        return pdf_vals

    def acceptance_ratio(
        self,
        new_inputs,
        old_inputs,
        new_log_like,
        old_log_like,
        new_log_priors,
        old_log_priors,
        proposal_covariances,  # New parameter: covariance matrices for proposal distribution
        new_proposal_covariances,
    ):
        proposal_covariances += np.eye(proposal_covariances.shape[1]) * 1e-6

        new_proposal_covariances += np.eye(new_proposal_covariances.shape[1]) * 1e-6
        # Compute posterior probabilities
        old_log_post = self.evaluate_log_posterior(
            old_inputs, old_log_like, old_log_priors
        )
        new_log_post = self.evaluate_log_posterior(
            new_inputs, new_log_like, new_log_priors
        )

        # Compute proposal probability ratio: q(old|new) / q(new|old)
        # For symmetric proposals (like normal), this ratio = 1, but we'll compute it generally

        # Proposal probability: q(new|old) - probability of proposing new_inputs given old_inputs
        q_new_given_old = self.multivariate_normal_pdf_batch(
            new_inputs, old_inputs, proposal_covariances
        )

        # Proposal probability: q(old|new) - probability of proposing old_inputs given new_inputs
        q_old_given_new = self.multivariate_normal_pdf_batch(
            old_inputs, new_inputs, new_proposal_covariances
        )

        # Compute log proposal ratio (safer numerically)
        eps = np.finfo(float).eps  # Machine epsilon (~2.22e-16)
        # Or use a slightly larger value: eps = 1e-300

        q_new_given_old = np.maximum(q_new_given_old, eps)
        q_old_given_new = np.maximum(q_old_given_new, eps)

        # print("qold", q_old_given_new)
        # print("qnew", q_new_given_old)
        log_proposal_ratio = np.log(q_old_given_new) - np.log(q_new_given_old)

        # Metropolis-Hastings acceptance ratio
        log_alpha = (new_log_post - old_log_post) + log_proposal_ratio

        # Convert to probability and reshape
        alpha = np.exp(np.minimum(0, log_alpha))  # min(1, exp(log_alpha))
        return alpha[:, 0].reshape(-1, 1)

    @rank_zero_output_only
    def get_rejections(self, acceptance_ratios):
        u = self.rng.uniform(0, 1, acceptance_ratios.shape)
        return acceptance_ratios < u

    def adapt_proposal_cov(self, cov, chain, idx, adapt_interval, adapt_delay):
        if self._is_adapt_iteration(adapt_interval, idx, adapt_delay):
            start = self._get_window_start(idx, adapt_delay, adapt_interval)
            end = idx + 1
            n_param = chain.shape[1]
            flat_chain = [chain[:, i, start:end].flatten() for i in range(n_param)]
            return np.cov(flat_chain)
        return cov

    def _initialize_probabilities(self, inputs):
        log_priors = self.evaluate_log_priors(inputs)
        self._check_log_priors_for_zero_probability(log_priors)
        log_like = self.evaluate_log_likelihood(inputs)
        return log_priors, log_like

    def _perform_mcmc_step(self, inputs, cov, log_like, log_priors, scale):
        new_inputs = self.proposal(inputs, cov)
        new_cov = self.compute_covariance(new_inputs) * scale
        new_log_priors = self.evaluate_log_priors(new_inputs)
        new_log_like = self._eval_log_like_if_prior_nonzero(new_log_priors, new_inputs)

        accpt_ratio = self.acceptance_ratio(
            new_inputs,
            inputs,
            new_log_like,
            log_like,
            new_log_priors,
            log_priors,
            cov,
            new_cov,
        )

        rejected = self.get_rejections(accpt_ratio)

        inputs = np.where(rejected, inputs, new_inputs)
        log_like = np.where(rejected, log_like, new_log_like)
        log_priors = np.where(rejected, log_priors, new_log_priors)
        cov = np.where(rejected[:, None], cov, new_cov)

        return inputs, log_like, log_priors, rejected, cov

    @staticmethod
    def _is_adapt_iteration(adapt_interval, idx, adapt_delay):
        if adapt_interval is None:
            return False
        surpassed_delay = idx >= adapt_delay
        is_adapt_iteration = (idx - adapt_delay) % adapt_interval == 0
        return surpassed_delay and is_adapt_iteration

    @staticmethod
    def _get_window_start(idx, adapt_delay, adapt_interval):
        if idx >= adapt_delay + adapt_interval:
            return adapt_delay + 1
        return max(adapt_delay - adapt_interval + 1, 1)

    def _check_log_priors_for_zero_probability(self, log_priors):
        if any(~self._row_has_nonzero_prior_probability(log_priors)):
            raise ValueError(
                "Initial inputs are out of bounds; " f"prior log prob = {log_priors}"
            )

    def _eval_log_like_if_prior_nonzero(self, log_priors, inputs):
        pos_rows = self._row_has_nonzero_prior_probability(log_priors)
        log_likes = np.zeros((log_priors.shape[0], 1))
        if inputs[pos_rows].size != 0:
            log_likes[pos_rows] = self.evaluate_log_likelihood(inputs[pos_rows])
        return log_likes

    @staticmethod
    def _row_has_nonzero_prior_probability(log_priors):
        return ~(log_priors == -np.inf).any(axis=1)

    @staticmethod
    def _ensure_psd_cov_and_do_chol_decomp(cov):
        """
        Higham NJ. Computing a nearest symmetric positive semidefinite matrix.
        Linear Algebra and its Applications. 1988 May;103(C):103-118.

        Code implementation: https://stackoverflow.com/a/63131250/4733085
        """
        try:
            return np.linalg.cholesky(cov)
        except:
            warnings.warn(
                "Covariance matrix is not positive semi-definite; "
                "forcing negative eigenvalues to zero and rebuilding "
                "covariance matrix."
            )
            eigval, eigvec = np.linalg.eigh(cov)
            eigval[eigval < 0] = 0
            cov = (eigvec @ np.diag(eigval)) @ eigvec.T
            cov += 1e-14 * np.eye(len(eigval))
            return np.linalg.cholesky(cov)
