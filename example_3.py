# Illustrate how to use DAPPER
# to obtain benchmarks for a range of experimental control variables,
# and plot the compiled results as functions of the control variable (xlabel),
# with each curve being one method configuration.
#
# Specifically, we will reproduce figure 6.6 from [1], accessible through 
# http://books.google.no/books?id=FtDZDQAAQBAJ&q=figure+6.6.
# The figure reveals the relative importance of localization and inflation in the EnKF.
# Ref[1]: Book: "Data Assimilation: Methods, Algorithms, and Applications"
#         by M. Asch, M. Bocquet, M. Nodet.
#
# Also demostrates:
#  - Parallelization (distribution of experiment settings).
#  - Result data saving, loading, plotting. 

from common import *

# Seed management. Notice below that the seed only varies
# between repetitions, not settings or configurations.
sd0 = seed_init(8) # Base random seed.

from   mods.Lorenz95.boc15loc import setup
import mods.Lorenz95.core as core

setup.t.BurnIn = 0.2 # Experiment duration
setup.t.T = 0.8 # Experiment duration

# Specify the control variable (i.e. the plotting xlabel) of interest.
SETTING = sys.argv[1] # command-line argument #1
#SETTING = 'N'

# Define range of the experiment control variable.
if SETTING == 'N': # Ensemble size.
  #settings = ccat(arange(2,20),[20, 22, 25, 30, 40, 50, 70, 100]) 
  settings = [20, 30]

if SETTING == 'F': # Model forcing
  settings = arange(3,20)

# Experiments duplication (random seeds will be varied).
settings = array(settings).repeat(1)

# If this script is run
# - with the second argument PARALLELIZE,
#   then this function will split the 'settings' array into 'nCore' batches,
#   and distribute each to a WORKER which run the rest of the script in parallel.
#   Important: see comment in distribute() about enforcing single-core use by numpy.
# - without a second argument: script is run as normal (no parallelization).
settings, save_path, rep_inds = distribute(__file__,sys.argv,settings,SETTING)


##############################
# DA Configurations
##############################
N = '?' if SETTING=='N' else 20

cfgs  = List_of_Configs()
cfgs += Climatology()                                               # Baseline method
cfgs += OptInterp()                                                 # Baseline method

for upd_a in ['PertObs','Sqrt']:                                    # Update (_a) forms: stoch, determ.
  cfgs += EnKF(upd_a,N                                            ) # Pure EnKF
  cfgs += EnKF(upd_a,N            ,infl=1.01                      ) # + fixed, post-inflation, good around N=50.
  cfgs += EnKF(upd_a,N            ,infl=1.05                      ) # + fixed, post-inflation, good around N=17.
  cfgs += EnKF(upd_a,N            ,infl=1.10                      ) # + fixed, post-inflation, good around N=16.
cfgs += EnKF_N(      N                       ,xN=1.5              ) # + adaptive (≈optimal) inflation
#cfgs += LETKF(       N,loc_rad=2                                  ) # + localization with radius=2
#cfgs += LETKF(       N,loc_rad='?'                                ) # + localization with ≈optimal radius(N)
#cfgs += iLEnKS('-N',N,loc_rad='?'           ,xN=1.5,iMax=1,Lag=1  ) # + localization and adaptive inflation.
#cfgs += iLEnKS('-N',N,loc_rad='?'           ,xN=1.5,iMax=4,Lag='?') # + iterations, localization and adaptive inflation.
# NB: Using Lag=0 for the iLEnKS is not supported. But the Lag=1 is not quite the filter.
# TODO: Implement spatialized inflation?


##############################
# Setters for the control variable
##############################
def adjust_osse(variable,S):
  if   variable == 'F': core.Force = S
  elif variable == 'N': pass
  else: raise ValueError("OSSE changes not defined for variable " + variable)

def adjust_cfg(C,variable,S):
  if variable == 'F':
    if getattr(C,'loc_rad',None)=='?': C = C.update_settings(loc_rad=L95_rad(S,core.Force))
  elif variable == 'N':
    if getattr(C,'N'      ,None)=='?': C = C.update_settings(      N=S)
    if getattr(C,'loc_rad',None)=='?': C = C.update_settings(loc_rad=L95_rad(S,core.Force))
    if getattr(C,'Lag'    ,None)=='?': C = C.update_settings(    Lag=L95_lag(S,core.Force))
  else: raise ValueError("Config changes not defined for variable " + variable)
  return C

# Most DA methods are approximate => Leeway exists
#  => Tuneable parameters exits. Here, we define some tuning settings.
def L95_rad(N,F):
  # Approximately fitted (for infl=1.0) variogram (Gaussian).
  r = 13*(1-exp(-(N/50)**2))
  r *= sqrt(8/F) # Not tuned at all!
  return r
def L95_lag(N,F):
  # Approximately tuned for iLEnKS and sak08 settings.
  if     N<=7 : return 1
  if 8 <=N<=15: return N-6
  if 16<=N    : return 10


##############################
# Assimilate
##############################
avrgs = np.empty((len(settings),1,len(cfgs)),dict)
# avrgs uses nRepeats=1 coz repetition is done by settings replication.

#stats = np.empty_like(avrgs)
# Lines with the 'stats' array are commented out so that the stat objects
# (which contain full time series and therefore might require significant memory)
# that the array would hold, instead get discarded after each loop iterate.

for iS,(S,iR) in enumerate(zip(settings,rep_inds)):
  with coloring(): print('\n'+"settings[",iS,'/',len(settings)-1,"] ",SETTING,': ',S,sep="")
  adjust_osse(SETTING,S)

  seed(sd0+iR)
  xx,yy = simulate(setup)

  for iC,C in enumerate(cfgs):
    C = adjust_cfg(C,SETTING,S)
    seed(sd0+iR)
    stat = C.assimilate(setup,xx,yy)
    #stats[iS,0,iC] = stat
    avrgs[iS,0,iC] = stat.average_in_time()

  print_averages(cfgs,avrgs[iS,0])

# Results saved in the format below is supported by DAPPER's ResultsTable, whose main purpose
# is to collect result data from parallelized (or otherwise independent) experiments.
np.savez(save_path,
    avrgs    = avrgs,            # 3D array of dicts whose fields are the averages.
    xlabel   = SETTING,          # The control variable tag (string).
    abscissa = settings,         # Abscissa (xticks) array.
    labels   = cfgs.gen_names()) # For legends, etc


##############################
# Results save/load/presentation
##############################
if 'WORKER' in sys.argv: sys.exit(0) # quit if script is running as worker.

# The rest of this script only uses saved data (=> could be a separate script):
R = ResultsTable(save_path)
# The presentation below could also be done for downloaded data (e.g. from parallelization):
#R = ResultsTable('data/example_3/MyRemoteHost/N_runX')

R = ResultsTable('data/example_3/johansen/N_run2') # 
#R = ResultsTable('data/example_3/P2720L/N_run1'); R.load('data/example_3/johansen/N_run1') # Localization VGs

# Print averages of a given field.
# The subcolumns show the number of repetitions, crashes and the 1-sigma conf.
with coloring(): print("Averages over experiment repetition:")
R.print_mean_field('rmse_a',1,1,cols=slice(0,2))

BaseLineMethods = R.split(lambda x: x in ['Climatology', 'OptInterp', 'Var3D','ExtKF'])

# Plot
fig, ax = plt.subplots()
R.plot_1d('rmse_a',)
check += [toggle_lines()]; plt.sca(ax)
BaseLineMethods.plot_1d('rmse_a',color='k')
ax.legend().remove();

# Adjust plot
if R.xlabel=='N':
  ax.loglog()
  ax.grid(True,'minor')
  xt = [2,3,4,6,8,10,15,20,25,30,40,50,70,100]
  yt = [0.1, 0.2, 0.5, 1, 2, 5]
  ax.set_xticks(xt); ax.set_xticklabels(xt)
  ax.set_yticks(yt); ax.set_yticklabels(yt)


# Discussion of results shown in figure:
# Localization more important than inflation? Both are necessary, neither sufficient.
# Note: the LETKF with infl=1 diverges even for large ensemble sizes.
#       But simply with infl=1.01 one obtains a much more reasonable benchmark curve.
#       However, the difference to **tuned** inflation (represented here by the EnKF-N)
#       is still very clear.

##

