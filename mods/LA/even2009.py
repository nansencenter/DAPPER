# A mix of Evensen'2009 and sakov'2008

# NB: Since there is no noise, and the system is stable,
#     the benchmarks obtained from this configuration
#     - go to zero as T-->infty
#     - are highly dependent on the initial error.

# Doc: Consider deeply the ensemble subspace,
# and the model's stability.

from common import *

from mods.LA.core import sinusoidal_sample, Fmat

m = 1000
p = 4
obsInds = equi_spaced_integers(m,p)

tseq = Chronology(dt=1,dkObs=5,T=300,BurnIn=-1)

#def step(x,t,dt):
  #return np.roll(x,1,axis=x.ndim-1)
Fm = Fmat(m,-1,1,tseq.dt)
def step(x,t,dt):
  assert dt == tseq.dt
  return x @ Fm.T

f = {
    'm': m,
    'model': step,
    'TLM'  : Fm,
    'noise': 0
    }

# In the animation, it can sometimes/somewhat occur
# that the truth is outside 3*sigma !!!
# Yet this is not so implausible because sinusoidal_sample()
# yields (multivariate) uniform (random numbers) -- not Gaussian.
wnum  = 25
X0 = RV(sampling_func = lambda N: \
    sqrt(5)/10 * sinusoidal_sample(m,wnum,N))

@atmost_2d
def hmod(E,t):
  return E[:,obsInds]
def yplot(y):
  lh = plt.plot(obsInds,y,'g*',MarkerSize=8)[0]
  plt.pause(0.8)
  return lh

H = zeros((p,m))
for i,j in enumerate(obsInds):
  H[i,j] = 1.0

h = {
    'm': p,
    'model': hmod,
    'TLM'  : lambda x,t: H,
    'noise': GaussRV(C=0.01*eye(p)),
    'plot' : yplot,
    }
 
other = {'name': os.path.relpath(__file__,'mods/')}

setup = OSSE(f,h,tseq,X0,**other)



####################
# Suggested tuning
####################

# Not carefully tuned
#cfg.N         = 100
#cfg.infl      = 1.02
#cfg.AMethod   = 'PertObs'
#cfg.rot       = False
#cfg.da_method = EnKF

# Expected rmse_a = 0.3
#cfg.N         = 30
#cfg.infl      = 3.4
#cfg.AMethod   = 'PertObs'
#cfg.rot       = False
#cfg.da_method = EnKF

# NB: Note how inflation is not necessary
# for good rmse performance.
