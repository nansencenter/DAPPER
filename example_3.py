# TODO: grab top/bottom comments from old example_3

from dapper import *

sd0 = set_seed(8,init=True)


##############################
# Hidden Markov Model
##############################
from   dapper.mods.Lorenz96.bocquet2015loc import HMM
import dapper.mods.Lorenz96.core as core

HMM.t.BurnIn = 5
HMM.t.T = 10*HMM.t.BurnIn

HMM.param_setters = dict(
        F = lambda x: setattr(core,'Force',x)
)


##############################
# DA Configurations
##############################
cfgs = ExperimentList(unique=True)

for N in ccat(arange(5,10), arange(10, 20, 2), arange(20, 55, 5)):
    for rot in [True,False]:
        for infl in 1+array([0, .01, .02, .04, .07, .1, .2, .4, .7, 1]):
            for R in round2([a*b for b in [.1, 1, 10] for a in [1, 2, 4, 7]]):
                cfgs += Climatology()
                cfgs += OptInterp()
                cfgs += EnKF   ('PertObs', N, infl=infl, rot=rot            )
                cfgs += EnKF   ('Sqrt',    N, infl=infl, rot=rot            )
                cfgs += EnKF_N (           N, infl=infl, rot=rot            )
                cfgs += LETKF  (           N, infl=infl, rot=rot, loc_rad=R )


# Replicate all cfgs accross some more control variables
xps = ExperimentList()
for seed in range(4):  # Experiment repetitions
    for F in [8]:      # L96 Forcing param
        for xp in deepcopy(cfgs):
            xp.seed = seed
            xp.HMM_F = F
            xps += xp


##############################
# Run experiments
##############################
savepath = save_dir(__file__)
xps.assimilate(HMM,sd=sd0,mp=False,savepath=savepath)


##############################
# Plotting
##############################
# The following **only** uses saved data => Can run as a separate script.
savepath = os.path.join(dirs['data'],"ex3","P2721L","run_6")
xps_orig = load_xps(savepath)

# Remove experiments we don't want to plot:
xps = ExperimentHypercube([xp for xp in xps_orig if True
    and getattr(xp,'upd_a'    ,None)!="PertObs"
    and getattr(xp,'da_method',None)!="EnKF_N"
    and getattr(xp,'HMM_F')         !=10
    ])

# Single out a few particular experiment types to add to plot
xps.single_out(dict(da_method='EnKF' ,infl=1.0), 'No-INFL No-LOC' , ('infl'))
xps.single_out(dict(da_method='LETKF',infl=1.0), 'No-INFL'        , ('infl'))

# Plot
plt.ion()
fig, axs, hypercube, plot_data = plot1d(xps, 'N', 'rmse.a', fignum=1,
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
