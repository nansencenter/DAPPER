# ## Plot the results generated in example `basic_3a`

# #### Imports

import matplotlib.pyplot as plt
import numpy as np

import dapper as dpr
from dapper.xp_process import (default_fig_adjustments, default_styles,
                               discretize_cmap, make_label, rel_index)

# #### Load

# Paths
save_as = dpr.rc.dirs.data / "basic_3"
# save_as /= "run_2020-11-11__20-36-36"
save_as /= dpr.find_latest_run(save_as)

# Load
xps = dpr.load_xps(save_as)
xp_dict = dpr.xpSpace.from_list(xps)


# #### Plot

# Single out (highlight) particular settings, to add as a line to the plot.
#
# Note: Must use `infl=1.01` (not 1) to reproduce "no infl" scores in Ref[1],
# as well as rot=True (better scores can be obtained without rot).

highlight = xp_dict.label_xSection
highlight('NO-infl'    , ('infl'), da_method='LETKF', infl=1.01, rot=True)
highlight('NO-infl/loc', ('infl'), da_method='EnKF' , infl=1.01, rot=True)

# Choose attribute roles for plot

tunable = {'loc_rad', 'infl', 'xB', 'rot'}
axes = dict(outer="F", inner="N", mean="seed", optim=tunable)
# xp_dict.print("rmse.a", axes, subcols=False)  # as in basic_3a.py


# Define linestyle rules

def get_style(coord):
    S = default_styles(coord, True)
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
    xSect = getattr(coord, "xSect", False)
    if str(xSect).startswith("NO-"):
        S.ls = "--"
        S.marker = None
        S.label = xSect
    return S


# Plot

tables = xp_dict.plot('rmse.a', axes, get_style, title2=save_as)
default_fig_adjustments(tables)
plt.pause(.1)


# #### Plot with color gradient

# Remove experiments we don't want to plot here

xps = [xp for xp in xps if getattr(xp, "xSect", None) == None]
xp_dict = dpr.xpSpace.from_list(xps)

# Setup mapping: `loc_rad --> color gradient`

graded = "loc_rad"
axes["optim"] -= {graded}
grades = xp_dict.tickz(graded)
# cmap, sm = discretize_cmap(cm.Reds, len(grades), .2)
cmap, sm = discretize_cmap(plt.cm.rainbow, len(grades))


def get_style_with_gradient(coord):
    S = get_style(coord)
    if coord.da_method == "LETKF":
        grade = rel_index(getattr(coord, graded), grades, 1)
        S.c = cmap(grade)
        S.marker = None
        S.label = make_label(coord, exclude=[graded])
    return S


# Plot

# +
tables = xp_dict.plot('rmse.a', axes, get_style_with_gradient, title2=save_as)
default_fig_adjustments(tables)

# Colorbar
cb = tables.fig.colorbar(sm, ax=tables[-1].panels[0], label=graded)
cb.set_ticks(np.arange(len(grades)))
cb.set_ticklabels(grades)

plt.pause(.1)
# -

# #### Excercise:
# Make a `get_style()` that works well with `graded = "infl"`.
