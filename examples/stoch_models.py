# ## Similar to examples/stoch_model1.py,
# TODO: how does it differ from it?

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


def setup(hmm, xp):
    """Set attrs. of `hmm` as specified by `xp`, run truth simulation."""
    # TODO: Should do seed management?
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
                xp = da.EnKF('PertObs', N=100)
                xp.resoltn = resolution
                xp.stepper = step_kind
                xp.Diffus1 = diffusion_stddev
                xp.ObsErr2 = obs_noise_variance
                xps.append(xp)

# #### Run experiments
save_as = xps.launch(HMMs(), __file__, setup=setup)

# #### Load -- this block is redundant if in the same script as generates the data
# save_as = dpr.rc.dirs.data / "stoch_models"
# save_as /= dpr.find_latest_run(save_as)
# xps = dpr.load_xps(save_as)

# print(dpr.xpList(xps).tabulate_avrgs())  # flat/long/list print

# #### Print in wide form
xp_dict = dpr.xpSpace.from_list(xps)
axes = dict(outer="resoltn", inner="ObsErr2", mean="seed", optim={'Diffus1'})
xp_dict.print("rmse.a", axes, subcols=False)

# #### Plot
tables = xp_dict.plot('rmse.a', axes, title2=save_as)
plt.pause(.1)
