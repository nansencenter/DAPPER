# TODO: grab top/bottom comments from old example_3

# Alternative source: Figure 5.7 of
# http://cerea.enpc.fr/HomePages/bocquet/teaching/assim-mb-en.pdf

from dapper import *

# sd0 = set_seed(8,init=True)
sd0 = set_seed()


##############################
# Hidden Markov Model
##############################
from   dapper.mods.Lorenz96.bocquet2015loc import HMM
import dapper.mods.Lorenz96.core as core

HMM.t.BurnIn = 250
HMM.t.KObs = 10**5

HMM.param_setters = dict(
        F = lambda x: setattr(core,'Force',x)
)


##############################
# DA Configurations
##############################
cfgs = ExperimentList(unique=True)

# for N in ccat(arange(5,10), arange(10, 20, 2), arange(20, 55, 5)):
    # for infl in 1+array([0, .01, .02, .04, .07, .1, .2, .4, .7, 1]):
        # for R in round2([a*b for b in [.1, 1, 10] for a in [1, 2, 4, 7]]):
            # for rot in [False,True]:
                # cfgs += Climatology()
                # cfgs += OptInterp()
                # cfgs += EnKF   ('PertObs', N, infl=infl, rot=rot            )
                # cfgs += EnKF   ('Sqrt',    N, infl=infl, rot=rot            )
                # cfgs += EnKF_N (           N, infl=infl, rot=rot            )
                # cfgs += LETKF  (           N, infl=infl, rot=rot, loc_rad=R )

cfgs += EnKF   ('Sqrt',    50, infl=1.0, rot=True            )

# Replicate all cfgs accross non-da_method control variables
xps = ExperimentList()
for seed in range(8): # Experiment repetitions
    for F in [8]:      # L96 Forcing param
        for xp in deepcopy(cfgs):
            xp.seed = seed
            xp.HMM_F = F
            xps += xp


##############################
# Run experiments
##############################
savepath = xps.launch(HMM,sd0,True,__file__)

# xps = ExperimentList(load_xps(savepath))
# xps.print_avrgs()
# sys.exit(0)

##############################
# Plot results
##############################
# The following **only** uses saved data => Can run as a separate script.
xps = load_xps(savepath)

# Remove experiments we don't want to plot:
xps = [xp for xp in xps if True
    and getattr(xp,'upd_a'    ,None)!="PertObs"
    and getattr(xp,'da_method',None)!="EnKF_N"
    and getattr(xp,'HMM_F')         !=10
    ]

# Associate each control variable with a dimension in "hyperspace"
xps = ExperimentHypercube(xps)

# Single out a few particular experiment types to add to plot
xps.single_out(dict(da_method='EnKF' ,infl=1.0), 'No-INFL No-LOC' , ('infl'))
xps.single_out(dict(da_method='LETKF',infl=1.0), 'No-INFL'        , ('infl'))

# Plot
plt.ion()
fig, axs, hypercube, plot_data = plot1d(xps, 'N', 'rmse.a', # fignum=1,
         mean_axs=('seed',), # could also include 'rot' for example
        optim_axs=('loc_rad','rot','infl'),
    # Try uncommenting one (or more if non-conflict!) of these lines:
    #     marker_ax="da_method", color_ax='infl', color_in_legend=False, 
    #     marker_ax='seed',                      marker_in_legend=False,
    #  linestyle_ax="rot",                    linestyle_in_legend=True,
    #      panel_ax='seed',
        )


# Other plotting perspectives:
# fig, axs, hypercube, plot_data = plot1d(xps, 'infl', 'rmse.a', fignum=1,
#          mean_axs=('seed',),
#         optim_axs=('loc_rad','rot'),
#          color_ax='N', color_in_legend=False, 
#         # Try toggling these two lines on/off:
#          panel_ax='da_method',
#         # marker_ax='da_method',
#         )

# Other plotting perspectives:
# fig, axs, hypercube, plot_data = plot1d(xps, 'N', 'rmse.a', fignum=1,
#         panel_ax='infl',
#          mean_axs=('seed',),
#         optim_axs=('loc_rad','rot'),
#         )

fig.suptitle("\n".join([fig._suptitle.get_text(), os.path.basename(savepath)]))
##
