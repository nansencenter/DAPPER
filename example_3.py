"""Illustrate usage of DAPPER to run MANY benchmark experiments,

for a bunch of control variables,
and plot the compiled results as functions of the control variable.

Specifically, we will reproduce figure 6.6 from Ref[1].
The figure reveals the (relative) importance (in the EnKF) of
localization and inflation.

The code also demonstrates:
 - Parallelization (accross independent experiments) with mp=True/"GCP".
 - Data management with xpSpace: load, sub-select, print, plot.

Note: unless you have a lot of CPUs available, you probably want to reduce
      the number of experiments (by shortening the list of ``seed``
      or some other variable) in the code below.

Ref[1]: Book: "Data Assimilation: Methods, Algorithms, and Applications"
        by M. Asch, M. Bocquet, M. Nodet.
        Preview:
        http://books.google.no/books?id=FtDZDQAAQBAJ&q=figure+6.6
        Alternative source: "Figure 5.7" of:
        cerea.enpc.fr/HomePages/bocquet/teaching/assim-mb-en.pdf
"""

from dapper import *
sd0 = set_seed(3)

##############################
# Hidden Markov Model
##############################
from   dapper.mods.Lorenz96.bocquet2015loc import HMM
import dapper.mods.Lorenz96.core as core

# This is shorter than Ref[1], but we also use repetitions.
HMM.t.BurnIn = 5
HMM.t.KObs = 10**3

HMM.param_setters = dict(
        F = lambda x: setattr(core,'Force',x)
)


##############################
# DA Configurations
##############################
cfgs = xpList(unique=True)

for N in ccat(arange(5,10), arange(10, 20, 2), arange(20, 55, 5)):
    for infl in 1+array([0, .01, .02, .04, .07, .1, .2, .4, .7, 1]):
        for R in round2([a*b for b in [.1, 1, 10] for a in [1, 2, 4, 7]]):
            for rot in [False,True]:
                cfgs += Climatology()
                cfgs += OptInterp()
                cfgs += EnKF   ('PertObs', N, infl=infl, rot=rot            )
                cfgs += EnKF   ('Sqrt',    N, infl=infl, rot=rot            )
                cfgs += EnKF_N (           N, infl=infl, rot=rot            )
                cfgs += LETKF  (           N, infl=infl, rot=rot, loc_rad=R )

# Replicate all cfgs accross non-da_method control variables
xps = xpList()
for seed in range(8): # Experiment repetitions
    for F in [8,10]:  # L96 Forcing param
        for xp in deepcopy(cfgs):
            xp.seed = seed
            xp.HMM_F = F
            xps += xp


##############################
# Run experiments
##############################
savepath = xps.launch(HMM,sd0,True,__file__)


##############################
# Plot results
##############################
# The following **only** uses saved data
# => Can run as a separate script, where savepath is manually set.
# For example, I have result-data stored at:
# savepath = '~/dpr_data/example_3/run_2020-01-02_00-00-00'
# savepath = '~/dpr_data/example_3/run_2020-01-09_17-45-34'

# Load
xps = load_xps(savepath)

# Remove experiments we don't want to print/plot at the moment:
xps = [xp for xp in xps
    if  getattr(xp,'upd_a'    ,None) != "PertObs"
    and getattr(xp,'da_method',None) != "EnKF_N"
    and 6 <= getattr(xp,'HMM_F',nan) <= 9
    ]

# Associate each control variable with a dimension in "hyperspace"
xp_dict = xpSpace.from_list(xps)

# Single-out certain settings
# Note: Must use infl=1.01 (not 1) to reproduce Ref[1]'s figure,
# as well as rot=True (better scores obtainable w/o rot).
separate = xp_dict.label_cross_section
separate('NO-infl'     , ('infl'), da_method='LETKF', infl=1.01, rot=True)
separate('NO-infl-loc' , ('infl'), da_method='EnKF' , infl=1.01, rot=True)

## 
axes_allotment=dict(inner="N", mean="seed", optim=('loc_rad','infl','rot'))

# Print
xp_dict.print("rmse.a", {**axes_allotment, "outer":"da_method"}, subcols=False)

# Plot -- try moving the axes around the allotment
# (as suggested by the lines that are commented out).
plt.ion()
tabulated_data = xp_dict.plot('rmse.a', axes_allotment, # fignum=1,
    #     marker_axis="da_method", color_axis='infl', color_in_legend=False, 
    #     marker_axis='seed',                        marker_in_legend=False,
    #  linestyle_axis="rot",                      linestyle_in_legend=True,
         marker_axis='da_method',                   marker_in_legend=True,
    )
# Custom adjustments
beautify_fig_ex3(tabulated_data, savepath, xp_dict)

##
