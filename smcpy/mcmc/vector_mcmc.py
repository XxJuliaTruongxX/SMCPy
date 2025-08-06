import numpy as np
import warnings

from tqdm import tqdm

from ..log_likelihoods import Normal
from ..utils.mpi_utils import rank_zero_output_only
from scipy.spatial.distance import cdist, squareform, pdist


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

    def gamma_median_heuristic(self, Z, num_subsample=1000):
        """
        Computes the median pairwise distance in a random sub-sample of Z.
        Returns a \gamma for k(x,y)=\exp(-\gamma ||x-y||^2), according to the median heuristc,
        i.e. it corresponds to \sigma in k(x,y)=\exp(-0.5*||x-y||^2 / \sigma^2) where
        \sigma is the median distance. \gamma = 0.5/(\sigma^2)
        """
        inds = np.random.permutation(len(Z))[: np.max([num_subsample, len(Z)])]
        dists = squareform(pdist(Z[inds], "sqeuclidean"))
        median_dist = np.median(dists[dists > 0])
        sigma = np.sqrt(0.5 * median_dist)
        gamma = 0.5 / (sigma**2)

        return gamma

    def compute_covariance(self, inputs):
        gamma2 = 0.1
        D = inputs.shape[1]

        array_cov = np.zeros((len(inputs), D, D))
        kernel_sigma = 1.0 / self.gamma_median_heuristic(inputs)
        kernel_gamma = 1.0 / kernel_sigma
        for i, y in enumerate(inputs):
            R = gamma2 * np.eye(D)

            y_2d = y.reshape(1, -1)
            Z = np.array(inputs)
            if len(Z) > 0:
                sq_dists = cdist(y_2d, Z, "sqeuclidean")
                k = np.exp(-kernel_gamma * sq_dists)
                neg_differences = Z - y
                G = 2 * kernel_gamma * (k.T * neg_differences)

                step_size = 2.3**2 / 2
                G *= 2
                H = np.eye(len(Z)) - 1.0 / len(Z)
                R += step_size * G.T.dot(H.dot(G))

            L_R = np.linalg.cholesky(R)
            array_cov[i] = L_R @ L_R.T
        return np.array(array_cov)

    # num_samples is the number of mcmc steps
    def smc_metropolis(self, inputs, num_samples, cov):
        num_particles = inputs.shape[0]
        log_priors, log_like = self._initialize_probabilities(inputs)
        for i in range(num_samples):
            cov = self.compute_covariance(inputs)
            inputs, log_like, log_priors, rejected = self._perform_mcmc_step(
                inputs, cov, log_like, log_priors
            )
            num_accepted = num_particles - np.sum(rejected)

            if num_accepted < inputs.shape[0] * 0.3:
                cov = cov * 1 / 5
            if num_accepted > inputs.shape[0] * 0.7:
                cov = cov * 2

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
            inputs, log_like, log_priors, rejected = self._perform_mcmc_step(
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
        proposal_covariances=None,  # New parameter: covariance matrices for proposal distribution
    ):
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
            old_inputs, new_inputs, proposal_covariances
        )

        # Compute log proposal ratio (safer numerically)
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

    def _perform_mcmc_step(self, inputs, cov, log_like, log_priors):
        new_inputs = self.proposal(inputs, cov)
        new_log_priors = self.evaluate_log_priors(new_inputs)
        new_log_like = self._eval_log_like_if_prior_nonzero(new_log_priors, new_inputs)

        accpt_ratio = self.acceptance_ratio(
            new_inputs, inputs, new_log_like, log_like, new_log_priors, log_priors, cov
        )

        rejected = self.get_rejections(accpt_ratio)

        inputs = np.where(rejected, inputs, new_inputs)
        log_like = np.where(rejected, log_like, new_log_like)
        log_priors = np.where(rejected, log_priors, new_log_priors)

        return inputs, log_like, log_priors, rejected

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
