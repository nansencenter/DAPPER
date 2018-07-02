# 

##############################
# Setup
##############################
from common import *
plt.style.use('AdInf/paper2.mplstyle')
#mpl.rcParams['figure.subplot.hspace']= 0.4 # to have suplot titles
from AdInf.filters import *

sd0 = seed_init(14) # base random seed

from mods.LorenzUV.lorenz95 import setup_full, setup_trunc, LUV
#(nU=36,J=10,F=10,h=1,b=10,c=10)

T     = 500  # length (unitless time) of each experiment
dtObs = 0.15 # DAW
tF = setup_full .t; tF.T = T; tF.dkObs = round(dtObs/tF.dt)
tT = setup_trunc.t; tT.T = T; tT.dkObs = round(dtObs/tT.dt)
dk = validate_int(tT.dt / tF.dt)

CtrlVar = 'F'
S = 16
print_c(CtrlVar,S)

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

# ADAPTIVE INFLATION METHODS
for N in [20]: # NB N=15 is too small for F>12
  cfgs += EnKF_pre('Sqrt',N=N,infl=1.36)
  #cfgs += EnKF_N       (N=N,                                   name='EnKF-$N$')
  #cfgs += EnKF_N_mod   (N=N, L=None,    nu_f=5)
  cfgs += EAKF_A07     (N=N,           var_f=1e-2,             name='EAKF adaptive')
  #cfgs += EAKF_A07     (N=N, damp=1.0, var_f=1e-2)
  cfgs += ETKF_Xplct   (N=N, L=None,    nu_f=1e3,              name='ETKF adaptive nu_f=1e3')
  #cfgs += ETKF_Xplct   (N=N, L=1e3,      nu0=10,               name='ETKF adaptive nu_o=1, L=1e3')
  #cfgs += ETKF_Xplct   (N=N, L=1e3,      nu0=10, nu_o1=0,      name='ETKF adaptive nu_o=?, L=1e3')
  #cfgs += ETKF_InvCS   (N=N, L=1e3,      nu0=10, Uni=1,        name='ETKF InvCS Quad')
  #cfgs += ETKF_InvCS   (N=N, L=1e3,      nu0=10, Uni=1, Var=1, name='ETKF InvCS Var')
  #cfgs += ETKF_Xplct   (N=N, L=None,    nu_f=1e3, infl=1.015)
  #cfgs += ETKF_Xplct   (N=N, L=None,    nu_f=1e4)
  #cfgs += ETKF_mode    (N=N, L=None,    nu_f=1e4)
  #cfgs += EnKF_N_Xplct (N=N, L=None,    nu_f=1e4)
  #cfgs += EnKF_N_Xplct (N=N, L=None,    nu_f=1e3)
  #cfgs += EnKF_N_Xplct (N=N, L=None,    nu_f=1e3, Cond=0)
  cfgs += EnKF_N_Xplct (N=N, L=None,    nu_f=1e4, Cond=0,      name='EnKF-$N$ hybrid')

for c in cfgs:
  if not hasattr(c,'detp'): c.detp = 1

##############################
# Assimilate
##############################
avrgs = np.empty(len(cfgs),dict)
stats = np.empty_like(avrgs)

setattr(LUV,CtrlVar,S)
xx,yy = simulate_or_load('AdInf/bench_LUV.py', setup_full, sd0, CtrlVar+'='+str(S))
prmzt = estimate_parameterization(xx)

for iC,Config in enumerate(cfgs):
  seed(sd0)
  
  LUV.prmzt = prmzt[Config.detp]
  stat = Config.assimilate(setup_trunc,xx[::dk,:nU],yy)
  avrg = stat.average_in_time()

  stats[iC] = stat
  avrgs[iC] = avrg
print_averages(cfgs, avrgs,statkeys=
    ['rmse_a','rmv_a','infl','a','b'])


##############################
# Plot
##############################
cfgs.assign_names(do_tab=False,ow=False)

class trialContextManager:
    def __enter__(self): pass
    def __exit__(self, *args): return True
trial = trialContextManager()

#cii = [1,3,4,5,6]
cii = arange(0,len(cfgs))

######

fig, axs = plt.subplots(len(cii),2,sharex=True)
for i, (s,ax) in enumerate(zip(stats[cii],axs)):
  #ax[0].set_title(cfgs[cii[i]].name)
  ax[0].set_ylabel(cfgs[cii[i]].name,fontsize=7)
  with trial: ax[0].plot(s.a   ,label='a'   ,c='C4')
  with trial: ax[0].plot(s.infl,label='infl',c='C2')
  with trial: ax[0].plot(s.b   ,label='b'   ,c='C1')
  with trial: ax[0].hlines(1,0,len(s.infl))

  ax[0].plot(s.rmse.a,label='RMS Err'    , c='C3')
  ax[0].plot(s.rmv .a,label='RMS Spread' , c='C0')

  ax[0].set_ylim(0,1.5)

  #with trial: ax[0].legend()

  with trial: ax[1].plot(s.nu_f,label='nu_f',c='k')
  with trial: ax[1].plot(s.nu_o,label='nu_o',c='C9')

for ax in fig.axes:
    ax.callbacks.connect('xlim_changed', on_xlim_changed)


##

dk = 1
fig, axs = plt.subplots(len(cii),1,sharex=True)
for i, (s,ax) in enumerate(zip(stats[cii],axs)):
  #with trial: la, = ax.plot(s.a   ,label='a'   ,c='C4')
  with trial: li, = ax.plot(s.infl[::dk],label='infl',c='C2',lw=1.5)
  with trial: lb, = ax.plot(s.b   [::dk],label='b'   ,c='C8',lw=2)
  with trial:       ax.hlines(1,0,len(s.infl),lw=0.5)

  le, = ax.plot(s.rmse.a[::dk],label='RMS Err'    , c='C3')
  ls, = ax.plot(s.rmv .a[::dk],label='RMS Spread' , c='C0')

  ax.yaxis.tick_right()
  ax.set_ylim(0,1.5)
  ax.set_xlim(2500/dk+array([0,500])/dk)
  ax.set_yticks([0.5,1,1.5])

axs[0].hlines(cfgs[0].infl,0,len(s.infl),label='infl',color='C2',lw=1.5) # NB validate
axs[0].set_ylabel('ETKF\ntuned')
axs[1].set_ylabel('EAKF\nadaptive')
axs[2].set_ylabel('ETKF\nadaptive')
axs[3].set_ylabel('EnKF-$N$\nhybrid')
axs[3].set_xlabel('DA cycle ($k$)')

fig.legend([li,le,ls],['Inflation','RMS Error','RMS Spread'],ncol=3,loc=(0.08,0.90))

#fname = 'infl_series'
#plt.savefig('data/AdInf/figs/'+fname+'.eps')
#plt.savefig('data/AdInf/figs/'+fname+'.pdf')


##
fig, (ax1,ax2) = plt.subplots(nrows=2,sharex=True)

opts = {'normed':True, 'bins':30,'alpha':0.5}

iC   = 1
xx   = stats[iC].rmse.a
yy   = stats[iC].rmv .a
mask = xx>0.0
ax1.hist(xx[mask],**opts,label=str(iC))
ax2.hist(yy[mask],**opts,label=str(iC))

iC   = 3
xx   = stats[iC].rmse.a
yy   = stats[iC].rmv .a
mask = xx>0.0
ax1.hist(xx[mask],**opts,label=str(iC))
ax2.hist(yy[mask],**opts,label=str(iC))
ax1.legend()

ax2.legend()

##

##



