# Benchmarks with model error in the LorenzUV system.

from common import *
from AdInf.filters import *

sd0 = seed(11*100 + hostname_hash())

# Range of experimental settings
SETTING = 'c'
if SETTING == 'c': # time scale ratio
  settings = round2sigfig(CurvedSpace(0.01,100,0.9,40),nfig=2)
  #settings = [0.1, 1, 10, 20, 30, 40]
  #settings = [min(settings, key=lambda x:abs(x-60))]
elif SETTING == 'h': # coupling constant 
  settings = ccat([0.1, 0.3], arange(0.5,1.6,0.1), [1.7, 2.0])

settings, save_path = distribute(__file__,sys.argv,settings,SETTING)

# For each experiment:
nRepeat = 1 # number of repeats
T = 20     # length (unitless time)

# Polynom. order of determ. error parameterizt.
# Order 2,3,4 only really work around c=10.
ORDER = 1


##############################
# Setup
##############################

#from mods.LorenzUV.wilks05  import setup_full, setup_trunc, LUV
from mods.LorenzUV.lorenz95 import setup_full, setup_trunc, LUV
nX = LUV.nX # num of "X"-vars

tF = setup_full .t; tF.T = T
tT = setup_trunc.t; tT.T = T;
dk = validate_int(tT.dt / tF.dt)

##############################
# Coupling estimation
##############################

# Estimate linear (deterministic) parameterization of unresovled scales.
# See mods/LorenzUV/illust_parameterizations.py for more details.
# Yields "good enough" estimates for T>100.
# Little diff whether using dt of setup_trunc or setup_full.
def estimate_parameterization(xx,order):
    TC = xx[tF.mask_BI,:nX] # Truth cropped to: burn-in and "X"-vars 
    gg = np.zeros_like(TC)  # "Unresolved tendency"
    if True: # Estimate based on dt of setup_full
      dt_ = tF.dt
    else:    # Estimate based on dt of setup_trunc
      TC  = TC[::dk]                 
      dt_ = tT.dt
      
    with set_tmp(LUV,'prmzt',lambda t,x: 0): # No parameterization
      for k,x in enumerate(progbar(TC[:-1],desc='Paramzt')):
        Mod   = setup_trunc.f(x,np.nan,dt_)
        Diff  = Mod - TC[k+1]
        gg[k] = Diff/dt_

    pc = np.polyfit(TC.ravel(), gg.ravel(), deg=order)
    print("Parameterization polynomial coeff fit:",pc)
    return lambda t,E: np.poly1d(pc)(E)

# Yields coupling used by truth, except for intermediate RK4 stages
def true_coupling(xx):
  coupling = zeros(nX)
  # RK4 uses intermediate stages => use a range.
  kk  = lambda t: (t-.9*tF.dt < tF.tt) & (tF.tt < t+.9*tF.dt)
  hcb = LUV.h*LUV.c/LUV.b
  def inner(t,E):
    Y_vars = xx[kk(t),nX:]
    for i in range(nX):
      coupling[i] = hcb * np.mean(Y_vars[:,LUV.iiY[i]],0).sum()
    return coupling
  return inner

def simulate_or_load(S,sd):
    path = save_dir(rel_path(__file__)+'/sims/')
    path += 'setup={:s} {:s}={:.3g} h={:s} dt={:.3g} T={:.3g} sd={:d}'.format(
        os.path.splitext(os.path.basename(setup_full.name))[0],
        SETTING,S,socket.gethostname(),tF.dt,tF.T,sd)
    try:
      msg   = 'loaded from'
      data  = np.load(path+'.npz')
      xx,yy = data['xx'], data['yy']
    except FileNotFoundError:
      msg   = 'saved to'
      xx,yy = simulate(setup_full)
      np.savez(path,xx=xx,yy=yy)
    print('Truth and obs',msg,'\n',path)
    return xx,yy


##############################
# Configs Wilks
##############################
# cfgs  = List_of_Configs()
# 
# # BASELINES
# #cfgs += EnKF_N(N=20         ,name='FULL')
# #cfgs += EnKF_N(N=20,nu=2.0  ,name='FULL')
# cfgs += EnKF_N(N=40,nu=2.0  ,name='FULL')
# #cfgs += EnKF_N(N=40         ,name='CHEAT')
# cfgs += EnKF_N(N=40,nu=2.0  ,name='CHEAT')
# #cfgs += EnKF_N(N=40,nu=3.0  ,name='CHEAT')
# cfgs += Climatology()
# cfgs += Var3D()

# infls = round2(CurvedSpace(1,3,0.98,20),0.01)
# for infl in infls:
#   cfgs += EnKF_N(N=10,infl=infl)
# for infl in infls:
#   cfgs += EnKF('Sqrt',N=10,infl=infl)

# ADAPTIVE INFLATION METHODS

##############################
# Configs Lorenz
##############################
cfgs  = List_of_Configs()

# BASELINES
cfgs += EnKF_N(N=20,nu=1.0  ,name='FULL')
cfgs += EnKF_N(N=20,nu=2.0  ,name='FULL')
cfgs += EnKF_N(N=40,nu=1.0  ,name='FULL')
cfgs += EnKF_N(N=40,nu=2.0  ,name='FULL')
cfgs += EnKF_N(N=40,nu=3.0  ,name='FULL')
cfgs += EnKF_N(N=80,nu=2.0  ,name='FULL')
cfgs += EnKF_N(N=20,nu=1.0  ,name='CHEAT')
cfgs += EnKF_N(N=40,nu=1.0  ,name='CHEAT')
cfgs += EnKF_N(N=40,nu=2.0  ,name='CHEAT')
cfgs += EnKF_N(N=80,nu=1.0  ,name='CHEAT')
cfgs += Climatology()
cfgs += Var3D()

infls = round2(CurvedSpace(1,3,0.98,20),0.01)
for infl in infls:
  cfgs += EnKF_N(N=20,infl=infl)
for infl in infls:
  cfgs += EnKF('Sqrt',N=20,infl=infl)

# ADAPTIVE INFLATION METHODS


##############################
# Assimilate
##############################
avrgs = np.empty((len(settings),nRepeat,len(cfgs)),dict)

for iS,S in enumerate(settings):
  print_c('\n'+SETTING+' value: ', S)
  if   SETTING == 'c': LUV.c = S
  elif SETTING == 'h': LUV.h = S

  for iR in range(nRepeat):
    sd = seed(sd0+iR)

    xx,yy = simulate_or_load(S,sd)
    
    LUV.prmzt = estimate_parameterization(xx,ORDER)

    for iC,Config in enumerate(cfgs):
      seed(sd)
      
      # Case: DA should use full model
      if 'FULL' in getattr(Config,'name',''):
        stat = Config.assimilate(setup_full,xx,yy)
        avrg = stat.average_subset(range(nX))

      # Case: DA should use trunc model but gets coupling from truth
      elif 'CHEAT' in getattr(Config,'name',''):
        with set_tmp(LUV,'prmzt',true_coupling(xx)), set_tmp(setup_trunc,'t',tF):
          stat = Config.assimilate(setup_trunc,xx[:,:nX],yy)
          avrg = stat.average_in_time()

      # Case: DA uses trunc model with parameterization as set above
      else:
        stat = Config.assimilate(setup_trunc,xx[::dk,:nX],yy)
        avrg = stat.average_in_time()

      avrgs[iS,iR,iC] = avrg
    print_averages(cfgs, avrgs[iS,iR])
  print_c('Average over',nRepeat,'repetitions:')
  print_averages(cfgs,average_each_field(avrgs[iS],axis=0))


##############################
# Save
##############################
cfgs.assign_names(do_tab=False,ow='prepend')
cnames = [c.name for c in cfgs]
print("Saving to",save_path)
np.savez(save_path,avrgs=avrgs,abscissa=settings,labels=cnames)


