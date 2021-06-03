# ## Present the results generated in example basic_3a
#
import matplotlib.pyplot as plt
import numpy as np

import dapper as dpr

save_as = dpr.rc.dirs.data / "basic_3"
# save_as /= "run_2020-11-11__20-36-36"
save_as /= dpr.find_latest(save_as)

# +
# Load
xps = dpr.load_xps(save_as)

# Prints all
# dpr.xpList(xps).print_avrgs(statkeys=["rmse.a","rmv.a"])
# -

# Associate each control variable with a coordinate/dimension
xp_dict = dpr.xpSpace.from_list(xps)

# Single out (highlight) particular settings.
# Note: Must use infl=1.01 (not 1) to reproduce "no infl" scores in Ref[1],
#       as well as rot=True (better scores can be obtained without rot).
highlight = xp_dict.label_xSection
highlight('NO-infl'    , ('infl'), da_method='LETKF', infl=1.01, rot=True)
highlight('NO-infl/loc', ('infl'), da_method='EnKF' , infl=1.01, rot=True)

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
plt.pause(.1)

# #### Plot with color gradient

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


# +
# Plot
tables = xp_dict.plot('rmse.a', axes, get_style_with_gradient, title2=save_as)
dpr.default_fig_adjustments(tables)

# Colorbar
cb = tables.fig.colorbar(sm, ax=tables[-1].panels[0], label=graded)
cb.set_ticks(np.arange(len(grades)))
cb.set_ticklabels(grades)

plt.pause(.1)
# -

# #### Excercise:
# Make a `get_style()` that works well with `graded = "infl"`.
