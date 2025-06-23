SMCPy - **S**equential **M**onte **C**arlo with **Py**thon 
==========================================================================
[![Build](https://github.com/XxJuliaTruongxX/SMCPy/actions/workflows/tests.yml/badge.svg)](https://github.com/XxJuliaTruongxX/SMCPy/actions)

## Description
SMCPy is an open-source package for performing uncertainty quantification using
a parallelized sequential Monte Carlo sampler.

## Key Features
* Alternative to Markov chain Monte Carlo for Bayesian inference problems
* Unbiased estimation of marginal likelihood for Bayesian model selection
* Parallelization through either numpy vectorization or mpi4py

# Quick Start

## Installation
To install SMCPy, use pip.
```sh
pip install smcpy
```

## Overview
To operate the code, the user supplies a computational model built in Python
3.6+, defines prior distributions for each of the model parameters to be
estimated, and provides data to be used for probabilistic model calibration. SMC
sampling of the parameter posterior distribution can then be conducted with ease
through instantiation of a sampler class and a call to the sample() method.

The two primary sampling algorithms implemented in this package are MPI-enabled
versions of those presented in the following articles, respectively:
> Nguyen, Thi Le Thu, et al. "Efficient sequential Monte-Carlo samplers for Bayesian
> inference." IEEE Transactions on Signal Processing 64.5 (2015): 1305-1319.
[Link to Article](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7339702) | [BibTeX Reference](https://scholar.googleusercontent.com/scholar.bib?q=info:L7AZJvppx1MJ:scholar.google.com/&output=citation&scisdr=CgUT24-FENXorVVNYK0:AAGBfm0AAAAAXYJIeK1GJKW947imCXoXAkfc7yZjQ7Oo&scisig=AAGBfm0AAAAAXYJIeNYSGEVCrlauowP6jMwVMHB_blTp&scisf=4&ct=citation&cd=-1&hl=en)


> Buchholz, Alexander, Nicolas Chopin, and Pierre E. Jacob. "Adaptive tuning of
> hamiltonian monte carlo within sequential monte carlo." Bayesian Analysis
> 1.1 (2021): 1-27.
[Link to Article](https://projecteuclid.org/journals/bayesian-analysis/advance-publication/Adaptive-Tuning-of-Hamiltonian-Monte-Carlo-Within-Sequential-Monte-Carlo/10.1214/20-BA1222.full) | [BibTeX Reference](https://scholar.googleusercontent.com/scholar.bib?q=info:wkjyyAN3q3UJ:scholar.google.com/&output=citation&scisdr=CgUA1gUaENXokaHu_K0:AAGBfm0AAAAAYXbr5K0e7EUBTRYw-hgqrmqC-G0ghzIo&scisig=AAGBfm0AAAAAYXbr5FfqGNe5PbrfGSvhMKzBoUbwdXDH&scisf=4&ct=citation&cd=-1&hl=en)

The first is a simple likelihood tempering approach in which the tempering
sequence is fixed and user-specified
([FixedSampler](https://github.com/nasa/SMCPy/blob/8b7813106de077c80992ba37d2d85944d6cce40c/smcpy/samplers.py#L44)).
The second is an adaptive approach that chooses the tempering steps based on a
target effective sample size ([AdaptiveSampler](https://github.com/nasa/SMCPy/blob/8b7813106de077c80992ba37d2d85944d6cce40c/smcpy/samplers.py#L92)).

This software was funded by and developed under the High Performance Computing
Incubator (HPCI) at NASA Langley Research Center.

## Example Usage
Using SMCPy is broken up into 3 main parts:

1. Creating the Model
2. Generating Noisy Data
3. Providing Prior Distributions
4. Setting Up and Running SMCPy

This example usage is an abbreviated version of the [Simple Example Tutorial](./examples/simple_example/run_example_adaptive.ipynb).

We start by generating noisy data that fits a linear model, which is represented as: $$y_i = Ax_i + B + \epsilon_i$$ where $y_i$ is a single observation at $x_i$ and $\epsilon_i$ represents independent and identically distributed measurement noise. For the purposes of the demonstration, we'll generate synthetic data by adding zero-mean, normally distributed noise with standard deviation $\sigma=2$ to a known line, defined by $A=2$ and $B=3.5$ across $x_i = [0, 99]$ **(This will be referenced as our <u>"true line"</u> from here on out).** 

The goal will be to learn the posterior probability distribution over $A$ and $B$ conditioned on the available data.


Using SMCPy is broken up into 4 main parts:

1. Creating the Model
2. Generating Noisy Data
3. Providing Prior Distributions
4. Setting Up and Running SMCPy

For this example, applying each of the main parts is as follows:
```python
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time

from scipy.stats import uniform

from smcpy import AdaptiveSampler, VectorMCMC, VectorMCMCKernel
from smcpy.paths import GeometricPath
from smcpy.utils.noise_generator import generate_noisy_data
from smcpy.utils.plotter import *

# 1. Creating the Model
def eval_model(theta):
    time.sleep(0.1)  # artificial slowdown to show off progress bar
    a = theta[:, 0, None]
    b = theta[:, 1, None]
    return a * np.arange(100) + b

# 2. Generating Noisy Data
std_dev = 2
model_output = eval_model(np.array([[2, 3.5]]))
noisy_data = generate_noisy_data(model_output, std_dev)

# require phi=0.2 be included in adaptive sequence
path = GeometricPath(required_phi=0.2)

# 3. Providing Prior Distributions
priors = [uniform(0.0, 6.0), uniform(0.0, 6.0)]

# 4. Setting Up and Running SMCPy
vector_mcmc = VectorMCMC(eval_model, noisy_data, priors, std_dev)
mcmc_kernel = VectorMCMCKernel(vector_mcmc, ("a", "b"), path=path)

smc = AdaptiveSampler(mcmc_kernel)
step_list, mll_list = smc.sample(
    num_particles=500, num_mcmc_samples=5, target_ess=0.7
)

sns.pairplot(pd.DataFrame(step_list[-1].param_dict))
sns.mpl.pyplot.savefig("pairwise.png")
```
When visualizing the posterior distribution with seaborn pairplot, we can see with the scatter plots that each SMC sample is represented as a dot containing a potential $A$ and $B$ value used to create our true line. For the histograms, we see that the majority of the SMC samples are clustered when $A \approx 2$ and when $B \approx 3.5$. This similarly reflects our true values of when $A = 2$ and $B = 3.5$!

![Simple Tutorial Example](./examples/simple_example/pairwise.png)

More in-depth explanations can be found in the
[Simple Tutorial Example](./examples/simple_example/run_example_adaptive.ipynb). 

To run this model in parallel using MPI, the MCMC kernel just needs to be built with the
ParallelVectorMCMC class in place of VectorMCMC. More in-depth explanations can be found in the
[MPI example](./examples/mpi_example/run_adaptive_example.py).

Install From Source
-----

Clone the repo and move into the package directory:

```sh
git clone https://github.com/nasa/SMCPy.git
cd SMCPy
```

Install requirements necessary to use SMCPy:

```sh
pip install -r requirements.txt
```

Optionally, if you'd like to use the MPI-enabled parallel sampler, install the
associated requirements:

```sh
pip install -r requirements_optional.txt
```

Add SMCPy to your Python path. For example:

```sh
export PYTHONPATH="$PYTHONPATH:/path/to/smcpy"
```

Run the tests to ensure proper installation:

```sh
pytest tests
```

## Contributing
1.  Fork (<https://github.com/nasa/SMCPy/fork>)
2.  Create your feature branch (`git checkout -b feature/fooBar`)
3.  Commit your changes (`git commit -am 'Add some fooBar'`)
4.  Push to the branch (`git push origin feature/fooBar`)
5.  Create a Pull Request

# Development
NASA Langley Research Center <br /> 
Hampton, Virginia <br /> 

This software was funded by and developed under the High Performance Computing Incubator (HPCI) at NASA Langley Research Center. <br /> 

## Authors
* Patrick Leser
* Michael Wang

# License
Notices:
Copyright 2018 United States Government as represented by the Administrator of
the National Aeronautics and Space Administration. No copyright is claimed in
the United States under Title 17, U.S. Code. All Other Rights Reserved.
 
Disclaimers
No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF
ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED
TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR
FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR
FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE
SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN
ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS,
RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS
RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY
DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF
PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."
 
Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE
UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY
PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY
LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE,
INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S
USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE
UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY
PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR
ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS
AGREEMENT.
