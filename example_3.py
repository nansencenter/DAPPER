# Illustrate how to use DAPPER
# to obtain benchmarks for a range of experimental control variables,
# and plot the compiled results as functions of the control variable (xlabel),
# with each curve being one method configuration.
#
# Specifically, we will reproduce figure 6.6 from [1], accessible through 
# http://books.google.no/books?id=FtDZDQAAQBAJ&q=figure+6.6
# The figure reveals the relative importance of localization and inflation in the EnKF.
# Ref[1]: Book: "Data Assimilation: Methods, Algorithms, and Applications"
#         by M. Asch, M. Bocquet, M. Nodet.
#
# Also demonstrates:
#  - Parallelization (distribution of experiment settings (xticks)).
#  - Result data saving, loading, plotting. 

from dapper import *

# Seed management. Notice below that the seed only varies
# between repetitions, not xticks or configurations.
sd0 = seed_init(8) # Base random seed.

from   dapper.mods.Lorenz95.boc15loc import HMM
import dapper.mods.Lorenz95.core as core

HMM.t.T = 4**4.0

# Specify the control variable (i.e. the plotting xlabel) of interest.
CtrlVar = sys.argv[1] # For example, run script with: `python example_3.py N`
# CtrlVar = 'N'

# Define range of the experiment control variable.
if CtrlVar == 'N': # Ensemble size.
  xticks = ccat(arange(2,20),[20, 22, 25, 30, 40, 50, 70, 100]) 

if CtrlVar == 'F': # Model forcing
  xticks = arange(3,20)

# Experiments duplication (random seeds will be varied).
xticks = array(xticks).repeat(32)

# If this script is run
# - with the 2nd command-line argument being PARALLELIZE,
#   then this function will split the 'xticks' array into 'nCore' batches,
#   and distribute each to a WORKER which run the rest of the script in parallel.
#   Important: see comment in distribute() about enforcing single-core use by numpy.
# - without a second argument: script is run as normal (no parallelization).
xticks, save_path, rep_inds = distribute(__file__,sys.argv,xticks,CtrlVar)


##############################
# DA Configurations
##############################
N = '?' if CtrlVar=='N' else 20

cfgs  = List_of_Configs()
cfgs += Climatology()                                                 # Baseline method
cfgs += OptInterp()                                                   # Baseline method

for upd_a in ['PertObs','Sqrt']:                                      # Update (_a) forms: stoch, determ.
  cfgs += EnKF(upd_a,N                                              ) # Pure EnKF
  cfgs += EnKF(upd_a,N            ,infl=1.01                        ) # + fixed, post-inflation, good around N=50.
  cfgs += EnKF(upd_a,N            ,infl=1.05                        ) # + idem                 , good around N=17.
  cfgs += EnKF(upd_a,N            ,infl=1.10                        ) # + idem                 , good around N=16.
cfgs += EnKF_N(      N                                              ) # + adaptive (≈optimal) inflation.
cfgs += EnKF_N(      N                       ,xN=2                  ) # + idem, with 2x confidence in inflation hyper-prior.
cfgs += LETKF(       N,loc_rad=2                                    ) # + localization with radius=2.
cfgs += LETKF(       N,loc_rad='?'                                  ) # + localization with ≈optimal radius(N)
cfgs += LETKF(       N,loc_rad='?',infl='-N' ,xN=2                  ) # + idem, with adaptive (≈optimal) inflation.
cfgs += LETKF(       N,loc_rad='$',infl='-N' ,xN=2                  ) # + idem, with adaptive (≈optimal) inflation.
cfgs += iLEnKS('-N' ,N,loc_rad='?'           ,xN=2 ,nIter=4,Lag='?' ) # + iterations, localization and adaptive inflation.


##############################
# Setters for the control variable
##############################
def adjust_osse(variable,X):
  if   variable == 'F': core.Force = X
  elif variable == 'N': pass
  else: raise ValueError("OSSE changes not defined for variable " + variable)

def adjust_cfg(C,variable,X):
  if variable == 'F':
    if getattr(C,'loc_rad',None)=='?': C = C.update_settings(loc_rad=L95_rad(X,core.Force))
    if getattr(C,'loc_rad',None)=='$': C = C.update_settings(loc_rad=L95_rad(X,core.Force))
  elif variable == 'N':
    if getattr(C,'N'      ,None)=='?': C = C.update_settings(      N=X)
    if getattr(C,'loc_rad',None)=='?': C = C.update_settings(loc_rad=L95_rad(X,core.Force))
    if getattr(C,'loc_rad',None)=='$': C = C.update_settings(loc_rad=1.5*L95_rad(X,core.Force))
    if getattr(C,'Lag'    ,None)=='?': C = C.update_settings(    Lag=L95_lag(X,core.Force))
  else: raise ValueError("Config changes not defined for variable " + variable)
  return C

# Most DA methods are approximate => Leeway exists
#  => Tuneable parameters exits. Here, we define some tuning xticks.
def L95_rad(N,F):
  # Approximately fitted (for infl=1.0) variogram (Gaussian).
  r = 0.08 + 10*(1-exp(-(N/40)**2))**0.8
  r *= sqrt(8/F) # Not tuned at all!
  return r
def L95_lag(N,F):
  # Approximately tuned for iLEnKS and sak08 xticks.
  if     N<=7 : return 1
  if 8 <=N<=15: return N-6
  if 16<=N    : return 10


##############################
# Assimilate
##############################
avrgs = np.empty((len(xticks),1,len(cfgs)),dict)
# avrgs uses nRepeats=1 coz repetition is done by xticks replication.

# stats = np.empty_like(avrgs)
# Lines with the 'stats' array are commented out so that the stat objects
# (which contain full time series and therefore might require significant memory)
# that the array would hold, instead get discarded after each loop iterate.

for iX,(X,iR) in enumerate(zip(xticks,rep_inds)):
  with coloring(): print('\n'+"xticks[",iX,'/',len(xticks)-1,"] ",CtrlVar,': ',X,sep="")
  adjust_osse(CtrlVar,X)

  seed(sd0+iR)
  xx,yy = simulate(HMM)

  for iC,C in enumerate(cfgs):
    C = adjust_cfg(C,CtrlVar,X)
    seed(sd0+iR)
    stat = C.assimilate(HMM,xx,yy)
    # stats[iX,0,iC] = stat
    avrgs[iX,0,iC] = stat.average_in_time()

  print_averages(cfgs,avrgs[iX,0])

# Results saved in the format below is supported by DAPPER's ResultsTable, whose main purpose
# is to collect result data from parallelized (or otherwise independent) experiments.
np.savez(save_path,
    avrgs      = avrgs,            # 3D array of dicts, whose fields are the averages.
    xlabel     = CtrlVar,          # The control variable tag (string).
    xticks     = xticks,           # xticks (array).
    labels     = cfgs.gen_names()) # List of strings.


##############################
# Results load & presentation
##############################
if 'WORKER' in sys.argv: sys.exit(0) # quit if script is running as worker.

# The rest of this script only uses saved data ...
R = ResultsTable(save_path)
# ... => could be run as a separate script,
# on downloaded data (e.g. from parallelization):
# R = ResultsTable(dirs['data']+'/example_3/MyRemoteHost/N_runX')

# Print averages of a given field.
# The "subcolumns" show the number of repetitions, crashes and the 1-sigma conf.
with coloring(): print("Averages over experiment repetition:")
R.print_mean_field('rmse_a',1,1,cols=slice(0,2))

# Separate out the baseline methods from the rest
BaseLineMethods = R.split(lambda x: x in ['Climatology', 'OptInterp', 'Var3D','ExtKF'])

# Plot
fig, ax = plt.subplots()
R.plot_1d('rmse_a',)
# The commented-out lines make checkmarks that toggle on/off the curves.
# if 'checkmarks' not in locals(): checkmarks = []
# checkmarks += [toggle_lines()];
BaseLineMethods.plot_1d('rmse_a',color='k')

# Adjust plot
if R.xlabel=='N':
  ax.loglog()
  ax.grid(True,'minor')
  xt = [2,3,4,6,8,10,15,20,25,30,40,50,70,100]
  yt = [0.1, 0.2, 0.5, 1, 2, 5]
  ax.set_xticks(xt); ax.set_xticklabels(xt)
  ax.set_yticks(yt); ax.set_yticklabels(yt)



# Discussion of results shown in figure:
# - Localization more important than inflation? Both are necessary, neither sufficient.
# - Some rotation could be beneficial for the Sqrt EnKF's with N>30 (avoid worsening rmse).
# - The EnKFs with infl=1 (and no localization) diverge even for large ensemble sizes.
#   But simply with infl=1.01 one obtains a much more reasonable benchmark curve.
#   However, the difference to **tuned** inflation (represented here by the EnKF-N)
#   is still very clear.
# - TODO: replace tuning functions by argmin (i.e. R.minz_tuning). Merge with example_4.
#   Why: good tuning functions require too much manual labor, and bad ones yield noisy curves.
#   Also use 2d plotting tools.
# - TODO: more thought to localization + adaptive inflation
#   This should also enable {inflation+localization} to get more advantage over {localization only}.

##

