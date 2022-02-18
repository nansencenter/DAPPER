# ## Similar to examples/stoch_model1.py.
# This experiment instead runs a full range of observation error
# and diffusion settings for the model uncertainty, in order to
# demonstrate the effects of low-precision numerics, and when
# these can be used in a perfect random model.  Particularly,
# when the system is perturbed by a small noise process (low diffusion
# coeffient), the low-precision Euler-Maruyama scheme diverges, while
# the Runge-Kutta scheme maintains good filter performance with the same
# step size.  This is contrasted with the scenario in which low-precision
# numerics are used with a model high-uncertainty, in a noise driven model.
# When the diffusion coefficient is large, the difference between the DA with
# the two schemes is negligible. Furthermore, we find that when both
# schemes are in the high-precision configuration both have adedquate filter
# performance.  However, this is only for demonstration, as a step size on
# order 10^{-3} must be used in order to have the Euler-Maruyama scheme produce
# an adequate forecast across regimes. This level of precision is impractical
# for most realistic twin experiments, emphasizing the fact that the
# Runge-Kutta scheme is statistically robust even for step sizes of 10^{-2}.

# #### Imports
# <b>NB:</b> If you're on <mark><b>Gooble Colab</b></mark>,
# then replace `%matplotlib notebook` below by
# `!python -m pip install git+https://github.com/nansencenter/DAPPER.git` .
# Also note that liveplotting does not work on Colab.

# %matplotlib notebook
import matplotlib.pyplot as plt

import dapper as dpr
import dapper.da_methods as da

# #### Load experiment setup: the hidden Markov model (HMM)

from dapper.mods.Lorenz96s.grudzien2020 import HMMs

# set global pseudo-random seed for all experiments
seed = dpr.set_seed(3000)


def setup(hmm, xp):
    """Set attrs. of `hmm` as specified by `xp`, run truth simulation."""
    import dapper.mods.Lorenz96s as core

    core.diffusion = xp.Diffus1
    xx, yy = HMMs("Tay2", xp.resoltn, R=xp.ObsErr2).simulate()
    if xp.resoltn == "Low":
        xx = xx[::2]
    hmm = HMMs(xp.stepper, xp.resoltn, R=xp.ObsErr2)
    return hmm, xx, yy


# #### DA method and experiment listing

xps = dpr.xpList()
for resolution in ["Low", "High"]:
    for step_kind in ["RK4", "EM"]:
        for diffusion_stddev in [0.1, 0.25, 0.5, 0.75, 1.0]:
            for obs_noise_variance in [0.1, 0.25, 0.5, 0.75, 1.0]:
                xp = da.EnKF("PertObs", N=100)
                xp.resoltn = resolution
                xp.stepper = step_kind
                xp.Diffus1 = diffusion_stddev
                xp.ObsErr2 = obs_noise_variance
                xps.append(xp)

# #### Run experiments

save_as = xps.launch(HMMs(), __file__, setup=setup)

# #### Load data
# This block is redundant if running in the same script
# as generates the data and not using multiprocessing.

# +
# save_as = dpr.rc.dirs.data / "stoch_models"
# save_as /= dpr.find_latest_run(save_as)
# xps = dpr.load_xps(save_as)

# print(dpr.xpList(xps).tabulate_avrgs())
# Flat/long/list print
# -


# #### Print in wide form

xp_dict = dpr.xpSpace.from_list(xps)
dims = dict(outer="resoltn", inner="ObsErr2", optim={})
xp_dict.print("rmse.a", dims, subcols=False)

# #### Plot

tables = xp_dict.plot("rmse.a", dims, title2=save_as)
plt.pause(0.1)
