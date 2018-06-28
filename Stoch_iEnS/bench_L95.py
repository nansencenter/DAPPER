from common import *

sd0 = seed(3)

##############################
# DA Configurations
##############################
from mods.Lorenz95.boc15loc import setup
setup.t.KObs = 2000 # length (num of cycles) of each experiment

# Get experiment control variable (SETTING) from arguments
SETTING = sys.argv[1]
# Set range of experimental settings
if SETTING == 'dkObs': # time scale ratio
  settings = 1+arange(8)
  #settings = [min(settings, key=lambda x:abs(x-s)) for s in [10,12]]

settings = array(settings).repeat(2)

# Parallelization and save-path setup
settings, save_path, iiRep = distribute(__file__,sys.argv,settings,SETTING,nCore=12)


##############################
# Configs
##############################
cfgs  = List_of_Configs()

# BASELINES
cfgs += Climatology()
cfgs += Var3D()

iMax  = 5
LAG   = '0.4'
infls = round2sigfig(CurvedSpace(1,2,0.9,20),nfig=3)
xNs   = [1, 1.5, 2, 4]
#for N in [25]:
for N in [20, 25, 35, 50, 100]:
  ROT  = 1 if N>20 else 0
  for xN in xNs:
    cfgs += EnKF_N(        N=N,  xN=xN,  rot=ROT,                  )
    cfgs += iEnKS('-N'    ,N=N,  xN=xN,  rot=ROT, Lag=1  ,iMax=iMax)
    cfgs += iEnKS('-N'    ,N=N,  xN=xN,  rot=ROT, Lag=LAG,iMax=iMax)
  for infl in infls:
    cfgs += EnKF('PertObs',N=N,infl=infl,                          )
    cfgs += EnRML('NA'    ,N=N,infl=infl,         Lag=1  ,iMax=iMax)
    cfgs += EnRML(        ,N=N,infl=infl,         Lag=LAG,iMax=iMax)


##############################
# Assimilate
##############################
avrgs = np.empty((len(settings),1,len(cfgs)),dict)
stats = np.empty_like(avrgs)

for iS,(S,iR) in enumerate(zip(settings,iiRep)):
  print_c('\n'+SETTING,'value:', S,'index:',iS,'/',len(settings)-1)
  setattr(setup.t,SETTING,S)

  sd    = seed(sd0 + iR)
  xx,yy = simulate(setup)

  for iC,C in enumerate(cfgs):
    if isinstance(getattr(C,'Lag',None),str):
      # Set Lag (specified in cycles) to LAG (unitless time)
      Lag = int(round(float(C.Lag)/setup.t.dtObs))
      C = C.update_settings(Lag=Lag)

    seed(sd)

    stat = C.assimilate(setup,xx,yy)
    avrg = stat.average_in_time()

    stats[iS,0,iC] = stat
    avrgs[iS,0,iC] = avrg
  print_averages(cfgs, avrgs[iS,0],statkeys=['rmse_a','rmv_a','infl'])

#plot_time_series(stats[-1])


##############################
# Save
##############################
cfgs.assign_names(do_tab=False,ow='prepend')
cnames = [c.name for c in cfgs]
print("Saving to",save_path)
np.savez(save_path,avrgs=avrgs,abscissa=settings,labels=cnames)




# Archived:
#  In the loop:  set_infl(C,avrgs[iS,0])
# 
#  def set_infl(cfg,avrgs):
#  
#    if cfg._is(EnKF_N) or getattr(cfg,'upd_a',0)=='-N' or not hasattr(cfgs,'N'):
#      return cfg
#  
#    def find_iC(cond):
#      # Find index in cfgs cond 
#      iC = [i for i,C in enumerate(cfgs) if cond(C)]
#      assert len(iC)==1
#      return iC[0]
#  
#  
#    if cfg._is(EnKF):
#  
#      iC = find_iC(lambda C: C._is(EnKF_N) and C.N==cfg.N)
#      
#      if   cfg.upd_a=='Sqrt'   : a = 1;    b = 0
#      elif cfg.upd_a=='PertObs': a = 1.73; b = 0.23
#  
#    elif cfg._is(iEnKS) or cfg._is(EnRML):
#  
#      iC = find_iC(lambda C: C._is(iEnKS) and C.upd_a=='-N' and C.N==cfg.N and C.Lag==cfg.Lag)
#  
#      if   cfg._is(iEnKS): a = 1;    b = 0
#      elif cfg._is(EnRML): a = 1.73; b = 0.23
#  
#    infl = avrgs[iC]['infl'].val
#    infl = 1 + a*(infl-1) + b
#  
#    return cfg.update_settings(infl=infl)

