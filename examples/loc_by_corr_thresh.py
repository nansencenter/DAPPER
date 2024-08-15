# ## Test correlation thresholding as a means of adaptive correlation
# Refer to `basic_3` for more info on this script.

# #### Imports

# %matplotlib notebook
import matplotlib.pyplot as plt
import numpy as np

import dapper as dpr
import dapper.da_methods as da
import dapper.tools.viz as viz

# #### Hidden Markov Model

from dapper.mods.Lorenz96.sakov2008 import HMM  # isort:skip


def setup(hmm, xp):
    """Experiment init.: Set Lorenz-96 forcing. Seed. Simulate truth/obs."""
    import dapper as dpr  # req. on clusters
    import dapper.mods.Lorenz96 as core

    core.Force = xp.Force
    return dpr.seed_and_simulate(hmm, xp)


HMM.tseq.Ko = 10**3

# #### DA method configurations

# Param ranges
params = dict(
    xB=[0.1, 0.2, 0.4, 1],
    N=[4, 7, 10, 12, 14, 16, 20, 50],
    infl=1 + np.array([0, 0.01, 0.02, 0.04, 0.1, 0.2, 1]),
    # rot=[True, False],
    loc_rad=[1, 2, 4, 10, 40],
    thresh=[None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
)
for_params = dpr.combinator(params, seed=3000 + np.arange(5), Force=[8])

xps = dpr.xpList()
xps += for_params(da.Climatology)
xps += for_params(da.OptInterp)
xps += for_params(da.Var3D, B="eye")
xps += for_params(da.EnKF, upd_a="Sqrt")
xps += for_params(da.LETKF)
xps += for_params(da.LETKF_thresh)

# For reference
# xps += da.Climatology()                                     # 3.6
# xps += da.OptInterp()                                       # 0.95
# xps += da.Var3D(xB=0.02)                                    # 0.41
# xps += da.ExtKF(infl=6)                                     # 0.24
# xps += da.EnKF('Sqrt'   , N=28, infl=1.02, rot=True)        # 0.18
# xps += da.EnKF_N(         N=24, rot=True, xN=2)             # 0.18
# xps += da.iEnKS('Sqrt'  , N=40, infl=1.01, rot=True)        # 0.17
# xps += da.LETKF(          N=7 , infl=1.04, rot=True, loc_rad=4)  # 0.22
# xps += da.SL_EAKF(        N=7 , infl=1.07, rot=True, loc_rad=6)  # 0.23

# #### Run experiments
# Paralellize/distribute experiments across CPUs.

save_as = xps.launch(HMM, __file__, mp=True, setup=setup)


# #### Print results

# save_as = dpr.rc.dirs.data / "loc_by_corr_thresh"
# # save_as /= "run_2024-06-14__17-23-09"
# save_as /= dpr.find_latest_run(save_as)

# Load data
xps = dpr.load_xps(save_as)

# Print as a flat list (as in basic_2.py)
# print(dpr.xpList(xps).tabulate_avrgs(statkeys=["rmse.a","rmv.a"]))

# Associate each control variable with a "coordinate"
xp_dict = dpr.xpSpace.from_list(xps)

# Choose attribute roles for plot

tunable = {"loc_rad", "infl", "thresh"}
dims = dict(inner="N", mean="seed", optim=tunable)
# xp_dict.print("rmse.a", dims, subcols=True)

# Define linestyle rules


def get_style(coord):
    S = viz.default_styles(coord, True)
    if coord.da_method == "EnKF":
        S.c = "k"
    elif coord.da_method == "LETKF":
        S.c = "C1"
    elif coord.da_method == "LETKF_thresh":
        S.c = "C0"
    return S


# Plot

plt.ion()
tables = xp_dict.plot("rmse.a", dims, get_style, title2=save_as)
viz.default_fig_adjustments(tables)
plt.pause(0.1)
