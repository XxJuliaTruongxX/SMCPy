import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import uniform, norm
from smcpy import AdaptiveSampler, VectorMCMC, VectorMCMCKernel
from smcpy.utils.intervals import compute_intervals


class DampedHarmonicOscillator:
    def __init__(self, t):
        self.t = t

    def evaluate(self, params):
        C = params[:, 0, None]
        K = params[:, 1, None]
        term1 = 2 * np.exp(-C * self.t / 2)
        term2 = np.cos(np.sqrt(K - C**2 / 4) * self.t)
        return term1 * term2


def plot_data(t, y, y_noisy):
    plt.figure()
    plt.plot(t, y, "-k", label="True")
    plt.plot(t, y_noisy, "bx", label="Data", ms=4)
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement")
    plt.ylim(-1.5, 2.5)
    plt.xlim(0, 5)
    plt.savefig("data.png")


def plot_posterior_pairwise(step_list):
    sns.pairplot(pd.DataFrame(step_list[-1].param_dict))
    plt.savefig("pairplot.png")


def plot_intervals(t, cred_intervals, pred_intervals, t_data, data):
    plt.figure()
    plt.fill_between(
        t.flatten(), pred_intervals[0, :], pred_intervals[1, :], facecolor="0.8"
    )
    plt.fill_between(
        t.flatten(), cred_intervals[0, :], cred_intervals[1, :], facecolor="0.4"
    )
    plt.plot(t_data, data, "xb", label="Data", ms=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement")
    plt.ylim(-1.5, 2.5)
    plt.xlim(0, 5)
    plt.legend(["95% Prediction Interval", "95% Credible Interval", "Data"])
    plt.savefig("intervals.png")


if __name__ == "__main__":
    n = 501
    t = np.linspace(0, 5, n).reshape(1, -1)
    sigma = 0.1
    true_params = np.array([[1.5, 20.5]])
    data_seed = 1
    model = DampedHarmonicOscillator(t)
    priors = [uniform(0, 5), uniform(10, 30), uniform(0, 1)]
    interval_lvls = [0.025, 0.975]

    # Generate data
    y = model.evaluate(true_params)
    y_noisy = y + np.random.default_rng(data_seed).normal(0, sigma, n)
    np.savetxt("damped_harmonic_oscillator_data.csv", np.c_[t, y_noisy])
    plot_data(t, y, y_noisy)

    # Run SMC
    vmcmc = VectorMCMC(model.evaluate, y_noisy.reshape(1, -1), priors)
    kernel = VectorMCMCKernel(vmcmc, param_order=["C", "K", "std"])
    smc = AdaptiveSampler(kernel)
    step_list, nmll = smc.sample(num_particles=5000, num_mcmc_samples=5)
    plot_posterior_pairwise(step_list)

    # Get intervals
    samples = step_list[-1].params
    model.t = np.linspace(0, 5, 1000)
    out = model.evaluate(samples)
    noisy_out = out + norm(0, samples[:, -1]).rvs((out.T.shape)).T

    cred_intervals = compute_intervals(out, 0.05)
    pred_intervals = compute_intervals(noisy_out, 0.05)

    plot_intervals(model.t, cred_intervals, pred_intervals, t, y_noisy)
