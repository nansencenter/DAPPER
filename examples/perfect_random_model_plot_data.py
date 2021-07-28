# ## Present the results generated in perfect_random_model_experiment_data
#
import matplotlib.pyplot as plt
import dapper as dpr

save_as = dpr.rc.dirs.data / "perfect_random_rk"
save_as /= dpr.find_latest_run(save_as)


# +
# Load
xps = dpr.load_xps(save_as)

# Prints all
# dpr.xpList(xps).print_avrgs(statkeys=["rmse.a","rmv.a"])
# -

# Associate each control variable with a coordinate/dimension
xp_dict = dpr.xpSpace.from_list(xps)

# Print, with columns: `inner`. Also try setting `outer=None`.
axes = dict(outer="Diffusion", inner="ObsNoise")
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
