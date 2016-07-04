# A mix of Evensen'2009 and sakov'2008

from common import *

from mods.LA.fundamentals import X0pat, Fmat

m = 1000
p = 4
obsInds = equi_spaced_integers(m,p)

tseq = Chronology(dt=1,dkObs=5,T=300,BurnIn=-1)

def step(x,t,dt):
  return np.roll(x,1,axis=x.ndim-1)
#Fm = Fmat(m,-1,1,tseq.dt).T.copy()
#def step(x,t,dt):
  #assert dt == tseq.dt
  #return x @ Fm

f = {
    'm': m,
    'model': step,
    'noise': 0
    }

wnum  = 25

#% Forward noise
wnumQ     = 25
NQ        = 2*wnumQ+1
A         = X0pat(m,wnumQ,NQ)
A         = 1/10 * anom(A)[0] / sqrt(NQ)
Q         = A.T @ A
#% F.noise.chol = tsqrt(A*A',2*wnumQ)
#F.noise.chol = A.T


#X0 = RV(func = lambda N: X0pat(m,wnum,N))
#X0 = GaussRV(C = homogeneous_1D_cov(m,m/8,kind='Expo'):
X0 = GaussRV(C = 5*Q)


def hmod(E):
  if np.ndim(E) == 1:
    E = E.reshape((1,len(E)))
  N = len(E)
  hE = np.zeros((N,p))
  for n in range(N):
    hE[n,:] = E[n,obsInds]
  if len(hE) == 1:
    hE = hE.squeeze()
  return hE

h = {
    'm': p,
    'model': lambda x,t: hmod(x),
    'noise': GaussRV(C=0.01*eye(p))
    }
 
#other = {'name': os.path.basename(__file__)}
other = {'name': os.path.relpath(__file__,'mods/')}

params = OSSE(f,h,tseq,X0,**other)

####################
# Suggested tuning
####################
