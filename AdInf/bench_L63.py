# Benchmarks with model error in the Lorenz63 system.

##############################
# Setup
##############################
from common import *
from AdInf.filters import *

sd0 = seed_init(15) # base random seed

import mods.Lorenz63.core as L63
from   mods.Lorenz63.sak12 import setup
setup.t.dkObs = 15  # DAW
setup.t.T     = 500 # length (unitless time)

# Get experiment control variable (SETTING) from arguments
SETTING = sys.argv[1]

# Set range of experimental settings
if SETTING == 'Q': # Var of stoch error
  set_true  = lambda S: setattr(setup.f.noise,'C',CovMat(S*ones(setup.f.m)))
  set_false = lambda S: setattr(setup.f.noise,'C',0)
  settings  = round2sigfig(LogSp(1e-6,1e2,40),nfig=2)
  #settings = [1e-6, 1e-3, 0.1, 1]
  #settings = [min(settings, key=lambda x:abs(x-0.01))]

elif SETTING == 'FDA': # Forcing erroneously assumed by DA
  set_true  = lambda S: setattr(L63,'sig',10)
  set_false = lambda S: setattr(L63,'sig',S)
  settings  = arange(4,24)

elif SETTING == 'FTr': # Forcing used by truth
  set_true  = lambda S: setattr(L63,'sig',S)
  set_false = lambda S: setattr(L63,'sig',10)
  settings  = arange(4,24)

settings = array(settings).repeat(6)

# Parallelization and save-path setup
settings, save_path, iiRep = distribute(__file__,sys.argv,settings,SETTING,nCore=999)


##############################
# Configs
##############################
cfgs  = List_of_Configs()

# # BASELINES
# cfgs += Climatology()
# cfgs += Var3D()
# 
# cfgs += EnKF_N(N=3 , name='FULL', rot=False)
# cfgs += EnKF_N(N=80, name='FULL', rot=True)
# 
infls = round2(CurvedSpace(1,5,0.98,40),0.01)
for N in [3]:
  for infl in infls: cfgs += EnKF_pre('Sqrt',N=N,infl=infl)

# ADAPTIVE INFLATION METHODS
# for N in [3]:
#   cfgs += EAKF_A07     (N=N,           var_f=1e-2           )
#   cfgs += EAKF_A07     (N=N, damp=1.0, var_f=1e-2           )
#   cfgs += ETKF_Xplct   (N=N, L=None,    nu_f=1e3            )
#   cfgs += ETKF_Xplct   (N=N, L=None,    nu_f=1e3, infl=1.015)
#   cfgs += ETKF_Xplct   (N=N, L=None,    nu_f=1e4            )
#   cfgs += EnKF_N_Xplct (N=N, L=None,    nu_f=1e4            )
#   cfgs += EnKF_N_Xplct (N=N, L=None,    nu_f=0e3            )
#   cfgs += EnKF_N_mod   (N=N, L=None,    nu_f=5)
#   cfgs += EnKF_N_Xplct (N=N, L=None,    nu_f=1e3, Cond=False)
#   cfgs += EnKF_N_Xplct (N=N, L=None,    nu_f=1e4, Cond=False)



##############################
# Assimilate
##############################
avrgs = np.empty((len(settings),1,len(cfgs)),dict)
stats = np.empty_like(avrgs)

for iS,(S,iR) in enumerate(zip(settings,iiRep)):
  print_c('\n'+SETTING,'value:', S,'index:',iS,'/',len(settings)-1)
  set_true(S)

  sd    = seed(sd0 + iR)
  xx,yy = simulate_or_load(__file__, setup, sd, SETTING+'='+str(S))

  for iC,Config in enumerate(cfgs):
    seed(sd)
    
    if 'FULL' in getattr(Config,'name',''):
      set_true(S)
    else:
      set_false(S)

    stat = Config.assimilate(setup,xx,yy)
    avrg = stat.average_in_time()

    #stats[iS,0,iC] = stat
    avrgs[iS,0,iC] = avrg
  print_averages(cfgs, avrgs[iS,0],statkeys=
      ['rmse_a','rmv_a','infl','nu_a','a','b'])


##############################
# Save
##############################
cfgs.assign_names(do_tab=False,ow='prepend')
cnames = [c.name for c in cfgs]
print("Saving to",save_path)
np.savez(save_path,avrgs=avrgs,abscissa=settings,labels=cnames)



