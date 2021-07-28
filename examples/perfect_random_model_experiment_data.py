# ## Illustrate usage of DAPPER to (interactively) run a synthetic ("twin") experiment.
# This follows the perfect-random configuration discussed in `bib.grudzien2020numerical`

# #### Imports
# <b>NB:</b> If you're on <mark><b>Gooble Colab</b></mark>,
# then replace `%matplotlib notebook` below by
# `!python -m pip install git+https://github.com/nansencenter/DAPPER.git` .
# Also note that liveplotting does not work on Colab.

# %matplotlib notebook
import dapper as dpr
import dapper.da_methods as da
import dapper.mods as modelling
from dapper.mods.Lorenz96s.grudzien2020 import (EMDyn, RKDyn, TruthDyn, X0, Obs,
                                                ttruth_high_precision,
                                                ttruth_low_precision,
                                                tmodel_high_precision,
                                                tmodel_low_precision,
                                                Diffusions, SigmasR)

# #### Load experiment setup: the hidden Markov model (HMM)


# we define a hook function to reset experimental parameters
def setup_low_precision(hmm, xp):
    """Experiment init.: Set Lorenz96s diffusion, and observation uncertainty."""
    import dapper as dpr
    import dapper.mods.Lorenz96s as core

    # reset the operator attribute to the current parameter configuration
    core.Diffusion = xp.Diffusion

    # reconstruct the observation operator for each setting of obseravation precision
    Obs['noise'] = xp.ObsNoise
    dpr.set_seed(getattr(xp, 'seed', False))

    # generate the truth simulation
    HMM_truth = modelling.HiddenMarkovModel(TruthDyn, Obs, ttruth_low_precision, X0)
    xx, yy = HMM_truth.simulate()

    # by default, we use half the time step size for the ensemble simulation
    # as for the truth simulation
    xx = xx[::2]
    return xx, yy


def setup_high_precision(hmm, xp):
    """Experiment init.: Set Lorenz96s diffusion, and observation uncertainty."""
    import dapper as dpr
    import dapper.mods.Lorenz96s as core

    # reset the operator attribute to the current parameter configuration
    core.Diffusion = xp.Diffusion

    # reconstruct the observation operator for each setting of obseravation precision
    Obs['noise'] = xp.ObsNoise
    dpr.set_seed(getattr(xp, 'seed', False))

    # generate the truth simulation
    HMM_truth = modelling.HiddenMarkovModel(TruthDyn, Obs, ttruth_high_precision, X0)
    xx, yy = HMM_truth.simulate()

    return xx, yy


# #### Generate the same random numbers each time a simulation is run
seed = dpr.set_seed(3000)

# #### DA method configurations
# Param ranges
params = dict(
        # NOTE: WHY SHOULD THESE NOT BE LOADED HERE?
        # Diffusion = Diffusions,
        # ObsNoise = SigmasR,
        N = [100],
        rot = [False],
        infl = [1.0],
        seed = [seed],
)

# Combines all the params suitable for a method. Faster than "manual" for-loops.
xps = dpr.xpList()
for_params = dpr.combinator(params, Diffusion = Diffusions, ObsNoise= SigmasR)
xps += for_params(da.EnKF, upd_a='PertObs')

# #### Run experiments

### NOTE: Do we need to reset the random seed for each of the following experiments?

# define the separate ensemble simulations
mp = False     # 1 CPU only
scriptname = "perfect_random_low_precision"  # since __file__ does not work in Jupyter

HMM_em_low_precision_ensemble = modelling.HiddenMarkovModel(EMDyn, Obs,
                                                            tmodel_low_precision, X0)

HMM_rk_low_precision_ensemble = modelling.HiddenMarkovModel(RKDyn, Obs,
                                                            tmodel_low_precision, X0)


# run the Runge-Kutta ensemble and print results
save_as = xps.launch(HMM_rk_low_precision_ensemble,
                     scriptname + "_rk", mp, setup_low_precision)
print(xps.tabulate_avrgs())

# run the Euler-Maruyama ensemble and print results
save_as = xps.launch(HMM_em_low_precision_ensemble,
                     scriptname + "_em", mp, setup_low_precision)
print(xps.tabulate_avrgs())


scriptname = "perfect_random_high_precision"  # since __file__ does not work in Jupyter


HMM_em_high_precision_ensemble = modelling.HiddenMarkovModel(EMDyn, Obs,
                                                             tmodel_high_precision, X0)

HMM_rk_high_precision_ensemble = modelling.HiddenMarkovModel(RKDyn, Obs,
                                                             tmodel_high_precision, X0)


# run the Runge-Kutta ensemble and print results
save_as = xps.launch(HMM_rk_high_precision_ensemble,
                     scriptname + "_rk", mp, setup_high_precision)
print(xps.tabulate_avrgs())

# run the Euler-Maruyama ensemble and print results
save_as = xps.launch(HMM_em_high_precision_ensemble,
                     scriptname + "_em", mp, setup_high_precision)
print(xps.tabulate_avrgs())
