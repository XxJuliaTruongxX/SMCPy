from copy import copy
import numpy as np

from .kernel_base import MCMCKernel
from ..paths import GeometricPath

class VectorMCMCKernel(MCMCKernel):

    def __init__(self, vector_mcmc_object, param_order, path=None):
        self._mcmc = vector_mcmc_object
        self._param_order = param_order
        self._path = GeometricPath() if path is None else path

        self._mcmc.evaluate_log_posterior = self._path

    def mutate_particles(self, param_dict, num_samples, cov, phi):
        self._path.phi = phi
        param_array = self._conv_param_dict_to_array(param_dict)
        param_array, log_likes = self._mcmc.smc_metropolis(param_array,
                                                           num_samples,
                                                           cov)
        param_dict = self._conv_param_array_to_dict(param_array)
        return param_dict, log_likes

    def sample_from_prior(self, num_samples):
        param_array = self._mcmc.sample_from_priors(num_samples)
        return self._conv_param_array_to_dict(param_array)

    def sample_from_proposal(self, num_samples):
        param_array = self._path.proposal.rvs(num_samples)
        return self._conv_param_array_to_dict(param_array)

    def get_log_likelihoods(self, param_dict):
        param_array = self._conv_param_dict_to_array(param_dict)
        return self._mcmc.evaluate_log_likelihood(param_array)

    def get_log_priors(self, param_dict):
        param_array = self._conv_param_dict_to_array(param_dict)
        log_priors = self._mcmc.evaluate_log_priors(param_array)
        return np.sum(log_priors, axis=1).reshape(-1, 1)

    def _conv_param_array_to_dict(self, param_array):
        return dict(zip(self._param_order, param_array.T))

    def _conv_param_dict_to_array(self, param_dict):
        dim0 = 1
        if not isinstance(param_dict[self._param_order[0]], (int, float)):
            dim0 = len(param_dict[self._param_order[0]])
        param_array = np.zeros((dim0, len(self._param_order)))

        for i, k in enumerate(self._param_order):
            param_array[:, i] = param_dict[k]
        return param_array
