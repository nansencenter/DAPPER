# Benchmarks with model error in the LorenzUV system.

from common import *
from AdInf.filters import *

sd0 = seed(11*100 + hostname_hash())

# Range of experimental settings
SETTING = 'F'
if SETTING == 'c': # time scale ratio
  settings = round2sigfig(CurvedSpace(0.01,100,0.9,40),nfig=2)
  #settings = [0.1, 1, 10, 20, 30, 40]
  #settings = [min(settings, key=lambda x:abs(x-60))]
  infls    = round2(CurvedSpace(1,1.6,0.9,20),0.01)
elif SETTING == 'h': # coupling constant 
  settings = round2sigfig(CurvedSpace(0.01,10,0.9,40),nfig=2)
  infls    = round2(CurvedSpace(1,1.6,0.9,20),0.01)
elif SETTING == 'F': # coupling constant 
  settings = round2sigfig(CurvedSpace(1,60,0.9,40),nfig=2)
  infls    = round2(CurvedSpace(1,3,0.98,20),0.01)
  settings = [10]
  #infls    = round2(CurvedSpace(1,1.30,0.94,10),0.01)
elif SETTING == 'R': # coupling constant 
  settings = round2sigfig(LogSp(0.01,40,20),nfig=2)
  infls    = round2(CurvedSpace(1,3,0.98,20),0.01)


settings, save_path = distribute(__file__,sys.argv,settings,SETTING)

# For each experiment:
nRepeat = 1  # number of repeats
T = 200      # length (unitless time)

##############################
# Setup
##############################

#from mods.LorenzUV.wilks05  import setup_full, setup_trunc, LUV
from mods.LorenzUV.lorenz95 import setup_full, setup_trunc, LUV
nU = LUV.nU # num of "U"-vars

tF = setup_full .t; tF.T = T
tT = setup_trunc.t; tT.T = T;
dk = validate_int(tT.dt / tF.dt)

##############################
# Helper function
##############################

# Coupling estimation
# Estimate linear (deterministic) parameterization of unresovled scales.
# See mods/LorenzUV/illust_parameterizations.py for more details.
# Yields "good enough" estimates for T>100.
# There's little diff whether using dt of setup_trunc or setup_full.
# Polynom order 2,3,4 only really work around c=10.
def estimate_parameterization(xx):
    TC = xx[tF.mask_BI,:nU] # Truth cropped to: burn-in and "U"-vars 
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

    parameterizations = {}
    funcs = {} # keep dict due to python's 'late binding'
    for order in range(4):
      pc = np.polyfit(TC.ravel(), gg.ravel(), deg=order)
      funcs[order] = np.poly1d(pc)
      parameterizations[order] = lambda t,E,n=order: funcs[n](E)
    parameterizations[-99] = 'PLACEHOLDER'
    #print("Parameterization polynomial coeff fit:",pc)
    #fig, ax = plt.subplots()
    #ax.scatter(TC[::400].ravel(), gg[::400].ravel(),
         #facecolors='none', edgecolors=blend_rgb('k',0.5),s=40)
    #uu = linspace(*ax.get_xlim(),201)
    #ax.plot(uu,np.poly1d(pc)(uu),'r',lw=4.0)
    #plt.pause(0.2)
    return parameterizations

# Yields coupling used by truth, except for intermediate RK4 stages
def true_coupling(xx):
  coupling = zeros(nU)
  # RK4 uses intermediate stages => use a range.
  kk  = lambda t: (t-.9*tF.dt < tF.tt) & (tF.tt < t+.9*tF.dt)
  hcb = LUV.h*LUV.c/LUV.b
  def inner(t,E):
    Y_vars = xx[kk(t),nU:]
    for i in range(nU):
      coupling[i] = hcb * np.mean(Y_vars[:,LUV.iiY[i]],0).sum()
    return coupling
  return inner

def simulate_or_load(S,sd): 
    path = save_dir(rel_path(__file__)+'/sims/',pre=os.environ.get('SIM_STORAGE',''))
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
# Configs
##############################
cfgs  = List_of_Configs()

# BASELINES
# cfgs += EnKF_N(N=20,nu=1.0  ,name='FULL')
# cfgs += EnKF_N(N=20,nu=2.0  ,name='FULL')
# cfgs += EnKF_N(N=40,nu=1.0  ,name='FULL')
# cfgs += EnKF_N(N=40,nu=2.0  ,name='FULL')
# cfgs += EnKF_N(N=80,nu=1.0  ,name='FULL')
# cfgs += EnKF_N(N=80,nu=2.0  ,name='FULL')
# cfgs += EnKF_N(N=20,nu=1.0  ,name='CHEAT')
# cfgs += EnKF_N(N=20,nu=2.0  ,name='CHEAT')
# cfgs += EnKF_N(N=40,nu=1.0  ,name='CHEAT')
# cfgs += EnKF_N(N=40,nu=2.0  ,name='CHEAT')
# 
# cfgs += Climatology(detp=-99)
# cfgs += Var3D(detp=0)
# cfgs += Var3D(detp=1)
# 
# for order in [0,1]:
#   for N in [15,20]:
#     for infl in infls: cfgs += EnKF_N(     N=N,infl=infl,detp=order)
#     for infl in infls: cfgs += EnKF('Sqrt',N=N,infl=infl,detp=order)
# 
# # ADAPTIVE INFLATION METHODS
# vbs = round2sigfig(CurvedSpace(0.001,1.0,0.9,20),nfig=2)
# #vbs = round2sigfig(CurvedSpace(1e-5,1,1-1e-4,20),nfig=2)
# for N in [15,20]:
#   for vb in vbs: cfgs += ETKF_M11_v1(N=N,         v_b=vb,detp=1)
#   for vb in vbs: cfgs += EAKF_A07_v1(N=N,damp=0.9,v_b=vb,detp=1)

##############################
# Assimilate
##############################

avrgs = np.empty((len(settings),nRepeat,len(cfgs)),dict)
stats = np.empty_like(avrgs)
SKeys = ['rmse_a','rmv_a','infl','v_o']

for iS,S in enumerate(settings):
  print_c('\n'+SETTING+' value: ', S)
  if SETTING=='R':
    # Set obs cov
    R = CovMat(S*ones(setup_full.h.m))
    setup_full .h.noise.C = R
    setup_trunc.h.noise.C = R
  else:
    # Set model parameter
    setattr(LUV,SETTING,S)

  for iR in range(nRepeat):
    sd = seed(sd0+iR)

    xx,yy = simulate_or_load(S,sd)
    prmzt = estimate_parameterization(xx)

    for iC,Config in enumerate(cfgs):
      seed(sd)
      
      # Case: DA should use full model
      if 'FULL' in getattr(Config,'name',''):
        stat = Config.assimilate(setup_full,xx,yy)
        avrg = stat.average_subset(range(nU))

      # Case: DA should use trunc model but gets coupling from truth
      elif 'CHEAT' in getattr(Config,'name',''):
        with set_tmp(LUV,'prmzt',true_coupling(xx)), set_tmp(setup_trunc,'t',tF):
          stat = Config.assimilate(setup_trunc,xx[:,:nU],yy)
          avrg = stat.average_in_time()

      # Case: DA uses trunc model with parameterization
      else:
        LUV.prmzt = prmzt[Config.detp]
        stat = Config.assimilate(setup_trunc,xx[::dk,:nU],yy)
        avrg = stat.average_in_time()

      #stats[iS,iR,iC] = stat
      avrgs[iS,iR,iC] = avrg
    print_averages(cfgs, avrgs[iS,iR],statkeys=SKeys)
  print_c('Average over',nRepeat,'repetitions:')
  print_averages(cfgs,average_each_field(avrgs[iS],axis=0),statkeys=SKeys)


##############################
# Save
##############################
cfgs.assign_names(do_tab=False,ow='prepend')
cnames = [c.name for c in cfgs]
print("Saving to",save_path)
np.savez(save_path,avrgs=avrgs,abscissa=settings,labels=cnames)


