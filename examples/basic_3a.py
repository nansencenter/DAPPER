# ## Illustrate usage of DAPPER to run MANY benchmark experiments.
#
# Launch many experiments (to explore a bunch of control variables),
# and plot the compiled results as in a variety of ways.
#
# As an example, we will reproduce Figure 6.6 from reference [1].
# The figure reveals the (relative) importance (in the EnKF) of
# localization and inflation. The output of this script is shown here:
# https://github.com/nansencenter/DAPPER#highlights
#
# The code also demonstrates:
# - Parallelization (accross independent experiments) with mp=True/Google.
# - Data management with xpSpace: load, sub-select, print, plot.
#
# NB: unless you have access to the DAPPER cluster, you probably want to reduce
# the number of experiments by shortening the list of `seed`
# (and maybe those of some tuning parameters) and/or reducing `Ko`.
#
# [1]: Asch, Bocquet, Nodet:
#      "Data Assimilation: Methods, Algorithms, and Applications",
#      Figure 6.6. Alternatively, see figure 5.7 of
#      http://cerea.enpc.fr/HomePages/bocquet/teaching/assim-mb-en.pdf .
#

# #### Imports

# %matplotlib notebook
import numpy as np

import dapper as dpr
import dapper.da_methods as da

# #### Hidden Markov Model

from dapper.mods.Lorenz96.bocquet2015loc import HMM  # isort:skip


def setup(hmm, xp):
    """Experiment init.: Set Lorenz-96 forcing. Seed. Simulate truth/obs."""
    import dapper as dpr  # req. on clusters
    import dapper.mods.Lorenz96 as core

    core.Force = xp.F
    return dpr.seed_and_simulate(hmm, xp)


# This is shorter than Ref[1], but we also use repetitions (a seed list).
HMM.tseq.Ko = 10 ** 4

# #### DA method configurations

# Param ranges
params = dict(
    xB=[0.1, 0.2, 0.4, 1],
    N=[5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50],
    infl=1 + np.array([0, 0.01, 0.02, 0.04, 0.07, 0.1, 0.2, 0.4, 0.7, 1]),
    rot=[True, False],
    loc_rad=dpr.round2sigfig([a * b for b in [0.1, 1, 10] for a in [1, 2, 4, 7]], 2),
)
# Combines all the params suitable for a method. See doc for dpr.combinator.
for_params = dpr.combinator(params, seed=3000 + np.arange(10), F=[8, 10])

xps = dpr.xpList()
xps += for_params(da.Climatology)
xps += for_params(da.OptInterp)
xps += for_params(da.Var3D, B="eye")
xps += for_params(da.EnKF, upd_a="PertObs")
xps += for_params(da.EnKF, upd_a="Sqrt")
xps += for_params(da.EnKF_N, infl=1.0)
xps += for_params(da.LETKF)


# #### Run experiments

# Paralellize/distribute experiments across CPUs.
mp = False  # 1 CPU only
# mp = 7         # 7 CPUs (requires that you pip-installed DAPPER with [MP])
# mp = True      # All CPUs
# mp = "Google"  # Requires access to DAPPER cluster

scriptname = "basic_3"  # since __file__ does not work in Jupyter
save_as = xps.launch(HMM, scriptname, mp, setup=setup)


# #### Print results

# Load data
if mp:
    xps = dpr.load_xps(save_as)
# Print as a flat list (as in basic_2.py)
# print(dpr.xpList(xps).tabulate_avrgs(statkeys=["rmse.a","rmv.a"]))

# Associate each control variable with a "coordinate"
xp_dict = dpr.xpSpace.from_list(xps)

# Print, split into tables by `outer` (also try None), and columns by `inner`.
tunable = {"loc_rad", "infl", "xB", "rot"}
dims = dict(outer="F", inner="N", mean="seed", optim=tunable)
xp_dict.print("rmse.a", dims, subcols=False)
