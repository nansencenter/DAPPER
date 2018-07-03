# Benchmarks with model error in the LorenzUV system.

##############################
# Setup
##############################
from common import *
from AdInf.filters import *

sd0 = seed_init(14) # base random seed

from mods.LorenzUV.lorenz95 import setup_full, setup_trunc, LUV
#(nU=36,J=10,F=10,h=1,b=10,c=10)

T     = 500  # length (unitless time) of each experiment
dtObs = 0.15 # DAW
tF = setup_full .t; tF.dkObs = round(dtObs/tF.dt); tF.T = T; 
tT = setup_trunc.t; tT.dkObs = round(dtObs/tT.dt); tT.T = T; 
dk = validate_int(tT.dt / tF.dt)


# Get experiment control variable (CtrlVar) from arguments
CtrlVar = sys.argv[1]
# Set range of experimental xticks
if CtrlVar == 'c': # time scale ratio
  xticks = round2sigfig(CurvedSpace(0.01,40,0.925,20),nfig=2)
  xticks = np.sort(ccat(xticks,[14,15]))
  #xticks = [min(xticks, key=lambda x:abs(x-s)) for s in [10,12]]
elif CtrlVar == 'h': # coupling constant 
  xticks = round2sigfig(CurvedSpace(0.01,10,0.9,40),nfig=2)
  #xticks = [min(xticks, key=lambda x:abs(x-s)) for s in [0.2,0.5,1]]
elif CtrlVar == 'F': # coupling constant 
  xticks = round2sigfig(CurvedSpace(1,30,0.9,40),nfig=2)

xticks = array(xticks).repeat(32)

# Parallelization and save-path setup
xticks, save_path, iiRep = distribute(__file__,sys.argv,xticks,CtrlVar)


##############################
# Parameterization estimation
##############################
nU = LUV.nU # num of "U"-vars

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


##############################
# Configs
##############################
cfgs  = List_of_Configs()

# BASELINES
cfgs += Climatology(detp=-99)
cfgs += Var3D(detp=0)
cfgs += Var3D(detp=1)

for nu in [1]:
  cfgs += EnKF_N(N=20, name='FULL' )
  cfgs += EnKF_N(N=80, name='FULL' , rot=True)
  #cfgs += EnKF_N(N=20, name='CHEAT')
  #cfgs += EnKF_N(N=80, name='CHEAT', rot=True)

# infls = round2(CurvedSpace(1,5,0.98,40),0.01)
# for N in [20]:
#   #for infl in infls: cfgs += EnKF_N  (       N=N,infl=infl)
#   #for infl in infls: cfgs += EnKF_N  (xN=2,  N=N,infl=infl)
#   #for infl in infls: cfgs += EnKF    ('Sqrt',N=N,infl=infl)
#   for infl in infls: cfgs += EnKF_pre('Sqrt',N=N,infl=infl)

# ADAPTIVE INFLATION METHODS
for N in [20]: # NB N=15 is too small for F>12
  #cfgs += EnKF_N       (N=N)
  #cfgs += EnKF_N_mod   (N=N, L=None,    nu_f=5)
  cfgs += EAKF_A07     (N=N,           var_f=1e-2)
  #cfgs += EAKF_A07     (N=N, damp=1.0, var_f=1e-2)
  #cfgs += ETKF_Xplct   (N=N, L=None,    nu_f=1e3)
  cfgs += ETKF_Xplct   (N=N, L=None,    nu_f=1e3, infl=1.015)
  #cfgs += ETKF_Xplct   (N=N, L=None,    nu_f=1e4)
  #cfgs += ETKF_InvCS   (N=N, L=1e3,      nu0=10, Uni=1, Var=0)
  #cfgs += EnKF_N_Xplct (N=N, L=None,    nu_f=1e4)
  #cfgs += EnKF_N_Xplct (N=N, L=None,    nu_f=1e3)
  #cfgs += EnKF_N_Xplct (N=N, L=None,    nu_f=1e3, Cond=0)
  cfgs += EnKF_N_Xplct (N=N, L=None,    nu_f=1e4, Cond=0)


# TUNING RANGES
# for var_f in round2sigfig(CurvedSpace(1e-3,1e0,0.99,10),1):
#   cfgs += EAKF_A07  (N=20,           var_f=var_f)
# for nu_f in round2sigfig(CurvedSpace(10,1e4,0.99,10),1):
#   cfgs += ETKF_Xplct(N=20, L=None,    nu_f=nu_f)
# for nu_f in round2sigfig(CurvedSpace(1e1,1e5,0.99,10),1):
#   cfgs += EnKF_N_Xplct(N=20, L=None,  nu_f=nu_f)


for c in cfgs:
  if not hasattr(c,'detp'): c.detp = 1


##############################
# Assimilate
##############################
avrgs = np.empty((len(xticks),1,len(cfgs)),dict)
stats = np.empty_like(avrgs)

for iX,(X,iR) in enumerate(zip(xticks,iiRep)):
  print_c('\n'+CtrlVar,'value:', X,'index:',iX,'/',len(xticks)-1)
  setattr(LUV,CtrlVar,X)

  sd    = seed(sd0 + iR)
  xx,yy = simulate_or_load(__file__, setup_full, sd, CtrlVar+'='+str(X))
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

    stats[iX,0,iC] = stat
    avrgs[iX,0,iC] = avrg
  print_averages(cfgs, avrgs[iX,0],statkeys=
      ['rmse_a','rmv_a','infl','a','b'])


##
np.savez(save_path,avrgs=avrgs,xlabel=CtrlVar,xticks=xticks,labels=cfgs.gen_names())
##
R = ResultsTable(save_path)
R.plot_1d('rmse_a',)
##


