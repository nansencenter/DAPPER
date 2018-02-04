from common import *

# Test DiscretePosRV
#
# This test includes learning, and therefore reports the final estimates
# rather than the average of the estimates of repeated one-cycle experiments.

# Observe that it also has bias, for the same reasons as the 'Xplclt" estimators:
# the likelihood is approximate (neglecting uncertainty in \bar x and \bar B).
# I.e. consistency is not attained for K-->∞ but, again, for N-->∞.

from common import *
from AdInf.filters import *

sd0 = seed(3)

class DiscretePosRV():
  def __init__(self,sc=1.0,nu=5,L=np.inf,R=1e2,N=1001):
    """
    Non-normalized equi-spaced discrete representation
    of a postitive random variable.

    Inputs 'sc' and 'nu' specify the intial pdf, assumed inverse-Chi2.
    Note thay this also specifies the domain FOR ALL TIME,
    and hence the initial parameters should be sufficiently agnostic.
    """
    self._xx    = linspace(*InvChi2Filter(sc,nu).domain(R), N) # grid
    #self.log_pp = iChi2_logp(sc,nu, self._xx) # log pdf eval on _xx (not norm'd)
    self.log_pp = np.zeros_like(self._xx)

    self.forget = exp(-1/L)

  @property
  def mode (self): return self._xx[self.log_pp.argmax()]
  @property
  def pp   (self): return exp(self.log_pp) # pdf eval on _xx (not norm'd)
  @property
  def const(self): return sum(self.pp)
  @property
  def mean (self): return sum(               self._xx*self.pp)/self.const
  @property
  def var  (self): return sum((self._xx-self.mean)**2*self.pp)/self.const

  def nrmlz(self):
    "NB: This is for numerical stability. Not to make it sum to one."
    self.log_pp = self.log_pp - self.log_pp.max()
  
  def forecast(self,k=1):
    # "Forgetting exponent" works by "diffusing" (not really) the pdf.
    self.log_pp = self.log_pp * self.forget**k
    self.nrmlz()

  def log_update(self,log_lklhd):
    self.log_pp = self.log_pp + log_lklhd(self._xx)
    self.nrmlz()

  def __str__(self):
    return 'Positive RV with\n' +\
        '  mean: {:.5g}\n'.format(self.mean) +\
        '  var : {:.5g}'  .format(self.var)


lklhd = 'Uni'
K = 10**2
m = 15
N = 8
N1 = N-1
eN = (N+1)/N

# Bunch of arrays for holding results
arr  = lambda K: np.full(K, nan)
stat = Bunch(mean=arr(K),var=arr(K))

B = eye(m) # randcov(m) diag(1+arange(m)**2) 
R = eye(m) # randcov(m) diag(1+arange(m)**2)
R = CovMat(R)
B = CovMat(B)

b = ones(m)
C = CovMat(R.full + B.full)

# This is what infl^2 should estimate
s2 = 1.0

Beta = DiscretePosRV(sc=1.0,nu=5,L=np.inf,R=1e2,N=1001)

fig, (ax1, ax2) = plt.subplots(nrows=2,sharex=True)
KP = 10
dP = K//KP
Colrs = plt.cm.plasma(linspace(0, 1, KP))

for k in range(K):
  x  = b + randn(m)     @ B.Right
  hE = b + randn((N,m)) @ B.Right / sqrt(s2)
  y  = x + randn(m)     @ R.Right

  hx = mean(hE,0)
  Y  = hE-hx
  dy = y - hx

  dR    = dy @ R.sym_sqrt_inv.T
  YR    = Y  @ R.sym_sqrt_inv.T

  V,s,UT = svd0(YR) # could use tsvd for some methods?
  du     = UT @ dR

  if 'Uni' in lklhd:
    trHPHR    = trace(YR.T @ YR)/N1 # sum(s**2)/N1
    log_lklhd = lambda b2: Chi2_logp(m + trHPHR*b2, m, dR@dR)
  elif 'Mult' in lklhd:
    dgn_v     = diag_HBH_I(s/sqrt(N1),min(N,m))
    log_lklhd = lambda b2: diag_Gauss_logp(0, dgn_v(b2), du).sum(axis=1)

  if k%dP==0 and k/dP<KP:
    ax1.plot(Beta._xx, Beta.pp,     c=Colrs[k//dP])
    ax2.plot(Beta._xx, Beta.log_pp, c=Colrs[k//dP])

  Beta.forecast()
  Beta.log_update(log_lklhd)

  stat.mean[k] = Beta.mean
  stat.var [k] = Beta.var



D = AlignedDict()
for key,val in stat.items():
  D["Final value of "+key] = str(val[-1])
print(D)







