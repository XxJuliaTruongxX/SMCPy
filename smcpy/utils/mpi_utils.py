def rank_zero_output_only(func):
    """
    Wrapper function that detects whether mpi4py is available and forces all
    ranks to output from rank 0 only. Intended to be used as a decorator where
    mpi-enabled MCMC kernels are deployed.
    """
    def wrapper(self, *args, **kwargs):
        output = func(self, *args, **kwargs)

        try:
            if hasattr(self, '_comm'):
                output = self._comm.bcast(output, root=0)
            else:
                output = self._mcmc_kernel._mcmc._comm.bcast(output, root=0)

        except AttributeError:
            pass

        return output

    return wrapper
