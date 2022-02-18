# ## Plot the results generated in example `basic_3a`

# #### Imports

import matplotlib.pyplot as plt

import dapper as dpr
import dapper.tools.viz as viz


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
highlight("NO-infl", ("infl"), da_method="LETKF", infl=1.01, rot=True)
highlight("NO-infl/loc", ("infl"), da_method="EnKF", infl=1.01, rot=True)

# Choose attribute roles for plot

tunable = {"loc_rad", "infl", "xB", "rot"}
dims = dict(outer="F", inner="N", mean="seed", optim=tunable)
# xp_dict.print("rmse.a", dims, subcols=False)  # as in basic_3a.py


# Define linestyle rules


def get_style(coord):
    S = viz.default_styles(coord, True)
    if coord.upd_a == "PertObs":
        S.c = "C2"
    elif coord.upd_a == "Sqrt":
        S.c = "C1"
    elif coord.da_method == "LETKF":
        S.c = "C3"
    if coord.rot:
        S.marker = "+"
    if str(coord.xSect).startswith("NO-"):
        S.ls = "--"
        S.marker = None
        S.label = coord.xSect
    return S


# Plot

tables = xp_dict.plot("rmse.a", dims, get_style, title2=save_as)
viz.default_fig_adjustments(tables)
plt.pause(0.1)


# #### Plot with color gradient

# Remove experiments we don't want to plot here

xps = [xp for xp in xps if getattr(xp, "xSect", None) == None]
xp_dict = dpr.xpSpace.from_list(xps)

# Get gradation/cmap for loc_rad

graded = "loc_rad"
dims["optim"] -= {graded}
grades = xp_dict.tickz(graded)
cmap, cbar = viz.discretize_cmap(plt.cm.rainbow, len(grades))


def get_style_with_gradient(coord):
    S = get_style(coord)
    if coord.da_method == "LETKF":
        g = grades.index(getattr(coord, graded))
        S.c = cmap(g)
        S.marker = None
        S.label = viz.make_label(coord, exclude=[graded])
    return S


# Plot

tables = xp_dict.plot("rmse.a", dims, get_style_with_gradient, title2=save_as)
cb = cbar(tables[[-1]][0].panels[0], grades, label=graded)
viz.default_fig_adjustments(tables)
plt.pause(0.1)

# #### Excercise:
# Make a `get_style()` that works well with `graded = "infl"`.
