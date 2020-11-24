import matplotlib.pyplot as plt
import numpy as np

from mpi4py import MPI
from scipy.stats import uniform

from smcpy.mcmc.parallel_mcmc import ParallelMCMC
from smcpy.mcmc.vector_mcmc_kernel import VectorMCMCKernel
from smcpy import SMCSampler


def gen_noisy_data(eval_model, std_dev, plot=True):
    y_true = eval_model(np.array([[2, 3.5]]))
    noisy_data = y_true + np.random.normal(0, std_dev, y_true.shape)
    if plot and MPI.COMM_WORLD.Clone().Get_rank() == 0:
        plot_noisy_data(x, y_true, noisy_data)
    return noisy_data


def plot_noisy_data(x, y_true, noisy_data):
    fig, ax = plt.subplots(1)
    ax.plot(x.flatten(), y_true.flatten(), '-k')
    ax.plot(x.flatten(), noisy_data.flatten(), 'o')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()


if __name__ == '__main__':

    np.random.seed(200)

    x = np.linspace(1, 5, 100)
    def eval_model(theta):
        a = theta[:, 0, None]
        b = theta[:, 1, None]
        return a * x + b

    std_dev = 0.5
    noisy_data = gen_noisy_data(eval_model, std_dev, plot=False)

    # configure
    num_particles = 1000
    num_smc_steps = 20
    num_mcmc_samples = 10
    ess_threshold = 0.8
    priors = [uniform(0., 6.), uniform(0., 6.)]

    comm = MPI.COMM_WORLD.Clone()
    parallel_mcmc = ParallelMCMC(comm, eval_model, noisy_data, priors, std_dev)

    phi_sequence = np.linspace(0, 1, num_smc_steps)

    mcmc_kernel = VectorMCMCKernel(parallel_mcmc, param_order=('a', 'b'))
    smc = SMCSampler(mcmc_kernel)
    step_list, evidence = smc.sample(num_particles, num_mcmc_samples,
                                     phi_sequence, ess_threshold)

    print('mean vector = {}'.format(step_list[-1].compute_mean()))
