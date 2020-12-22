"""Illustrate usage of DAPPER to run MANY benchmark experiments.

Launch many experiments (to explore a bunch of control variables),
and plot the compiled results as in a variety of ways.

As an example, we will reproduce Figure 6.6 from reference [1].
The figure reveals the (relative) importance (in the EnKF) of
localization and inflation.

The code also demonstrates:
- Parallelization (accross independent experiments) with mp=True/Google.
- Data management with xpSpace: load, sub-select, print, plot.

NB: unless you have access to the DAPPER cluster, you probably want to reduce
the number of experiments by shortening the list of ``seed``
(and maybe those of some tuning parameters) and/or reducing ``KObs``.
Also, the resulting output can be previewed at
https://github.com/nansencenter/DAPPER#highlights

[1]: Asch, Bocquet, Nodet:
     "Data Assimilation: Methods, Algorithms, and Applications",
     https://books.google.no/books?id=FtDZDQAAQBAJ&q=figure+6.6 .
     Alternatively, see figure 5.7 of
     http://cerea.enpc.fr/HomePages/bocquet/teaching/assim-mb-en.pdf .
"""

import numpy as np
from matplotlib import pyplot as plt

import dapper as dpr
import dapper.da_methods as da

##############################
# Hidden Markov Model
##############################
from dapper.mods.Lorenz96.bocquet2015loc import HMM  # isort:skip


def setup(hmm, xp):
    """Experiment init.: Set Lorenz-96 forcing. Seed. Simulate truth/obs."""
    import dapper as dpr
    import dapper.mods.Lorenz96 as core
    core.Force = xp.F
    return dpr.seed_and_simulate(hmm, xp)


# This is shorter than Ref[1], but we also use repetitions (a seed list).
HMM.t.KObs = 10**4


##############################
# DA Configurations
##############################
# Param ranges
params = dict(
    xB       = [.1, .2, .4, 1],
    N        = [5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50],
    infl     = 1+np.array([0, .01, .02, .04, .07, .1, .2, .4, .7, 1]),
    rot      = [True, False],
    loc_rad  = dpr.round2sigfig([a*b for b in [.1, 1, 10] for a in [1, 2, 4, 7]], 2),
)
# Combines all the params suitable for a method. Faster than "manual" for-loops.
for_params = dpr.get_param_setter(params, seed=3000+np.arange(10), F=[8, 10])

xps = dpr.xpList()
xps += for_params(da.Climatology)
xps += for_params(da.OptInterp)
xps += for_params(da.Var3D, B="eye")
xps += for_params(da.EnKF, upd_a="PertObs")
xps += for_params(da.EnKF, upd_a="Sqrt")
xps += for_params(da.EnKF_N, infl=1.0)
xps += for_params(da.LETKF)


##############################
# Run experiments
##############################

# Paralellize/distribute experiments across CPUs.
mp = False     # 1 CPU only
# mp = 7         # 7 CPUs (requires that you pip-installed DAPPER with [MP])
# mp = True      # All CPUs
# mp = "Google"  # Requires access to DAPPER cluster

save_as = xps.launch(HMM, __file__, mp, setup)


##############################
# Present results
##############################
# The following "section" **only** uses saved data.
# => Can run as a separate script, by setting save_as manually, e.g.
# save_as = dpr.rc.dirs.data / "basic_3" / ""run_2020-11-11__20:36:36"

# Load
xps = dpr.load_xps(save_as)

# Prints all
# xpList(xps).print_avrgs(statkeys=["rmse.a","rmv.a"])

# Associate each control variable with a coordinate/dimension
xp_dict = dpr.xpSpace.from_list(xps)

# Single out (highlight) particular settings.
# Note: Must use infl=1.01 (not 1) to reproduce "no infl" scores in Ref[1],
#       as well as rot=True (better scores can be obtained without rot).
point_out = xp_dict.label_xSection
point_out('NO-infl', ('infl'), da_method='LETKF', infl=1.01, rot=True)
point_out('NO-infl/loc', ('infl'), da_method='EnKF', infl=1.01, rot=True)

# Print, with columns: `inner`. Also try setting `outer=None`.
tunable = {'loc_rad', 'infl', 'xB', 'rot'}
axes = dict(outer="F", inner="N", mean="seed", optim=tunable)
xp_dict.print("rmse.a", axes, subcols=False)


def get_style(coord):
    """Quick and dirty styling."""
    S = dpr.default_styles(coord, True)
    if coord.da_method == "EnKF":
        upd_a = getattr(coord, "upd_a", None)
        if upd_a == "PertObs":
            S.c = "C2"
        elif upd_a == "Sqrt":
            S.c = "C1"
    elif coord.da_method == "LETKF":
        S.c = "C3"
    if getattr(coord, "rot", False):
        S.marker = "+"
    Const = getattr(coord, "Const", False)
    if str(Const).startswith("NO-"):
        S.ls = "--"
        S.marker = None
        S.label = Const
    return S


# Plot
tables = xp_dict.plot('rmse.a', axes, get_style, title2=save_as)
dpr.default_fig_adjustments(tables)
plt.pause(1)


##############################
#  Plot with color gradient  #
##############################
# Remove experiments we don't want to plot here
xps = [xp for xp in xps if getattr(xp, "Const", None) == None]
xp_dict = dpr.xpSpace.from_list(xps)

# Setup mapping: loc_rad --> color gradient
graded = "loc_rad"
axes["optim"] -= {graded}
grades = xp_dict.tickz(graded)
# cmap, sm = dpr.discretize_cmap(cm.Reds, len(grades), .2)
cmap, sm = dpr.discretize_cmap(plt.cm.rainbow, len(grades))


def get_style_with_gradient(coord):
    S = get_style(coord)
    if coord.da_method == "LETKF":
        grade = dpr.rel_index(getattr(coord, graded), grades, 1)
        S.c = cmap(grade)
        S.marker = None
        S.label = dpr.make_label(coord, exclude=[graded])
    return S


# Plot
tables = xp_dict.plot('rmse.a', axes, get_style_with_gradient, title2=save_as)
dpr.default_fig_adjustments(tables)

# Colorbar
cb = tables.fig.colorbar(sm, ax=tables[-1].panels[0], label=graded)
cb.set_ticks(np.arange(len(grades)))
cb.set_ticklabels(grades)

plt.pause(1)

# Excercise:
# Make a get_style() that works well with graded = "infl".
