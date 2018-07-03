# Benchmarks with model error in the Lorenz95 system.

##############################
# Setup
##############################
from common import *
from AdInf.filters import *

sd0 = seed_init(14) # base random seed

import mods.Lorenz95.core as L95
from   mods.Lorenz95.sak08 import setup

setup.t.T = 500 # length (unitless time)
setup.t.dkObs = 3 # DAW


# Get experiment control variable (CtrlVar) from arguments
CtrlVar = sys.argv[1]

# Set range of experimental settings
if CtrlVar == 'Q': # Var of stoch error
  set_true  = lambda X: setattr(setup.f.noise,'C',CovMat(X*ones(setup.f.m)))
  set_false = lambda X: setattr(setup.f.noise,'C',0)
  xticks    = round2sigfig(LogSp(1e-6,2,20),nfig=2)
  #xticks   = [1e-6, 1e-3, 0.1, 1]
  #xticks   = [min(xticks, key=lambda x:abs(x-60))]

elif CtrlVar == 'FDA': # Forcing erroneously assumed by DA
  set_true  = lambda X: setattr(L95,'Force',8) # should also be in xticks!
  set_false = lambda X: setattr(L95,'Force',X)
  xticks    = arange(7,9+0.1,0.1)

elif CtrlVar == 'FTr': # Forcing used by truth
  set_true  = lambda X: setattr(L95,'Force',X)
  set_false = lambda X: setattr(L95,'Force',8) # should also be in xticks!
  xticks    = arange(7,9+0.1,0.1)

xticks = array(xticks).repeat(32)

# Parallelization and save-path setup
xticks, save_path, iiRep = distribute(__file__,sys.argv,xticks,CtrlVar)


##############################
# Configs
##############################
cfgs  = List_of_Configs()

# BASELINES
cfgs += Climatology()
cfgs += Var3D()

cfgs += EnKF_N(N=20, name='FULL', rot=False)
cfgs += EnKF_N(N=80, name='FULL', rot=True)

# infls = round2(CurvedSpace(1,5,0.98,40),0.01)
# for N in [20]:
#   for infl in infls: cfgs += EnKF_pre('Sqrt',N=N,infl=infl)

# ADAPTIVE INFLATION METHODS
for N in [20]:
  cfgs += EAKF_A07     (N=N,           var_f=1e-2           )
  cfgs += ETKF_Xplct   (N=N, L=None,    nu_f=1e3, infl=1.015)
  cfgs += EnKF_N_Xplct (N=N, L=None,    nu_f=1e4, Cond=False)



##############################
# Assimilate
##############################
avrgs = np.empty((len(xticks),1,len(cfgs)),dict)
stats = np.empty_like(avrgs)

for iX,(X,iR) in enumerate(zip(xticks,iiRep)):
  print_c('\n'+CtrlVar,'value:', X,'index:',iX,'/',len(xticks)-1)
  set_true(X)

  sd    = seed(sd0 + iR)
  xx,yy = simulate_or_load(__file__, setup, sd, CtrlVar+'='+str(X))

  for iC,Config in enumerate(cfgs):
    seed(sd)
    
    if 'FULL' in getattr(Config,'name',''):
      set_true(X)
    else:
      set_false(X)

    stat = Config.assimilate(setup,xx,yy)
    avrg = stat.average_in_time()

    #stats[iX,0,iC] = stat
    avrgs[iX,0,iC] = avrg
  print_averages(cfgs, avrgs[iX,0],statkeys=
      ['rmse_a','rmv_a','infl','nu_a','a','b'])


##
np.savez(save_path,avrgs=avrgs,xlabel=CtrlVar,xticks=xticks,labels=cfgs.gen_names())
##
R = ResultsTable(save_path)
R.plot_1d('rmse_a',)
##


