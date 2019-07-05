from dapper import *

sd0 = seed()

##############################
# DA Configurations
##############################
from dapper.mods.Lorenz95.sak08 import HMM
HMM.t.T = 120
HMM.t.dkObs = 8 # 4, 8, 12.
LAG = max(1,round(0.4 / HMM.t.dtObs))

# Get experiment control variable (CtrlVar) from arguments
CtrlVar = sys.argv[1]
# Set range of experimental settings
if CtrlVar == 'N': # ens size
  xticks = [13, 16, 17, 18, 20, 22, 25, 30, 40, 60, 100]
if CtrlVar == 'nIter': # max num of iterations
  xticks = [1, 2, 4, 8, 16, 32]

xticks = array(xticks).repeat(10)

# Parallelization and save-path setup
xticks, save_path, iiRep = distribute(__file__,sys.argv,xticks,CtrlVar,nCore=4)


##############################
# Configs
##############################
cfgs  = List_of_Configs(unique=True)

cfgs += Climatology()
cfgs += OptInterp()
cfgs += Var3D()

for N in [25]:
  if CtrlVar=='N': N = '?' # short-circuit
  for nIter in [3,10]:
    if CtrlVar=='nIter': nIter = '?' # short-circuit
    for infl in [1.01, 1.02, 1.04, 1.06, 1.10, 1.16, 1.25, 1.4]:
      for MDA in [False,True]:
        for rot in [False, True]:
          cfgs +=  EnKF('PertObs', N=N, infl=infl,                                       )
          cfgs +=  EnKF('Sqrt'   , N=N, infl=infl, rot=rot,                              )
          cfgs +=  EnKF('DEnKF'  , N=N, infl=infl, rot=rot,                              )
          cfgs += iEnKS('PertObs', N=N, infl=infl,          Lag=LAG, nIter=nIter, MDA=MDA)
          cfgs += iEnKS('Sqrt'   , N=N, infl=infl, rot=rot, Lag=LAG, nIter=nIter, MDA=MDA)
          cfgs += iEnKS('Order1' , N=N, infl=infl, rot=rot, Lag=LAG, nIter=nIter, MDA=MDA)


##############################
# Assimilate
##############################
avrgs = np.empty((len(xticks),1,len(cfgs)),dict)
stats = np.empty_like(avrgs)

for iX,(X,iR) in enumerate(zip(xticks,iiRep)):
  with coloring(): print('\n'+"xticks[",iX,'/',len(xticks)-1,"] ",CtrlVar,': ',X,sep="")
  # setattr(HMM.t,CtrlVar,X)

  sd    = seed(sd0 + iR)
  xx,yy = simulate(HMM)

  for iC,C in enumerate(cfgs):
    C.liveplotting = False

    if (CtrlVar in ['N', 'nIter']) and hasattr(C,CtrlVar):
      C = C.update_settings(**{CtrlVar:X})

    seed(sd)

    stat = C.assimilate(HMM,xx,yy)
    avrg = stat.average_in_time()

    # stats[iX,0,iC] = stat
    avrgs[iX,0,iC] = avrg
  print_averages(cfgs, avrgs[iX,0],statkeys=['rmse_a','rmv_a','infl'])

#plot_time_series(stats[-1])

np.savez(save_path,
    avrgs      = avrgs,
    xlabel     = CtrlVar,
    xticks     = xticks,
    tuning_tag = 'infl', 
    labels     = cfgs.gen_names(tab=True),
    meta       = {'dkObs':HMM.t.dkObs})


##############################
# Results load & presentation
##############################
if 'WORKER' in sys.argv: sys.exit(0) # quit if script is running as worker.

R = ResultsTable(save_path)

with coloring(): print("Averages over experiment repetition:")
R.print_mean_field('rmse_a',1,1,cols=slice(0,2))

BaseLineMethods = R.split(['Climatology', 'OptInterp', 'Var3D','ExtKF'])
BaseLineMethods.rm('Var3D')

R.rm("rot:0")
R.rm("EnKF")

# Plot
fig, ax = plt.subplots()
ax, ax2, _, _ = R.plot_1d_minz('rmse_a',)
ax.legend()
BaseLineMethods.plot_1d('rmse_a',color='k')

##

