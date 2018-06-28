##

# Test if an inflation-related statistic
# really has the distribution it's supposed to (Muihead Th 3.2.8).
# Also test related statistics.
#
# Note: Sometimes B     will play the role of barB of the paper,
#       while inv(barB) will play the role of B    of the paper.


## Preamble
from common import *
#seed(6)

from scipy.stats import chi2, invgamma, f
CS    = lambda loc, nu: chi2(df=nu,scale=loc/nu)
InvCS = lambda loc, nu: invgamma(a=nu/2,scale=loc*nu/2)
F     = lambda d1, d2, loc, scl: f(d1,d2,loc,scl)

K = 10**5   # Num of experiments
M = 3       # State length
N = 6       # Ens size
N1 = N-1
g  = N-M
eN = (N+1)/N

# True cov matrix
B = np.diag(1+np.arange(M))
#B = randcov(M)
iB = inv(B)
B12 = sqrtm(B)

# Init containers
num_u  = 5
uu     = [()]*num_u
stats  = OrderedDict((s, [zeros(K) for _ in range(num_u)]) for s in 'abcdefg')
BB = zeros((K,M,M))

for k in range(K):
  E     = B12@randn((M,N))            # \sim NormDist(0,B)
  A, bx = anom(E,1)
  barB  = A@A.T / N1                  # \sim    Wish( B,N1)
  ibB   = inv(barB)                   # \sim InvWish(iB,N1)
  BB[k] = barB

  # Vectors with which to make norm statistic
  uu[0] = ones(M)                               # fixed
  uu[1] = rand(M)                               # Uniform
  uu[2] = randn(M)                              # x (NormDist)
  uu[3] = B12@randn(M) - bx                     # x-bx
  uu[4] = B12@randn(M) / sqrt(chi2(g).rvs(1)/g) # t(0,B;g)

  for i,u in enumerate(uu):
    stats['a'][i][k] = (u@B@u) / (u@barB@u)
    stats['b'][i][k] = (u@barB@u) / (u@B@u)
    stats['c'][i][k] = (u@iB@u) / (u@ibB@u)
    stats['d'][i][k] = (u@iB@u)
    stats['f'][i][k] = (u@ibB@u)
    stats['g'][i][k] = (u@iB@u)

##

fig, axs = plt.subplots(nrows=num_u,sharex=True)

PVAR = 'g'
for i in range(num_u):
  upper = np.percentile(ccat(stats[PVAR]),99)
  xx    = linspace(1e-9,upper,201)

  axs[i].hist(stats[PVAR][i],bins=xx,alpha=0.9,normed=True)
  if PVAR=='a':
    axs[0].set_title('InvCS(1,N1)')
    lh, = axs[i].plot(xx, InvCS(loc=1,nu=N1).pdf(xx))
  elif PVAR=='b':
    axs[0].set_title('CS(1,N1)')
    lh, = axs[i].plot(xx,    CS(loc=1,nu=N1).pdf(xx))
  elif PVAR=='c':
    axs[0].set_title('CS((N-M)/N1,N-M)')
    lh, = axs[i].plot(xx,    CS(loc=(N-M)/N1, nu=N-M).pdf(xx))
  elif PVAR=='d':
    axs[0].set_title('CS(eN*M,M)')
    lh, = axs[i].plot(xx,    CS(loc=eN*M,nu=M).pdf(xx))
  elif PVAR=='f':
    axs[0].set_title('F')
    lh, = axs[i].plot(xx, F(M,N-M,0,eN*N1*M/(N-M)).pdf(xx))
  elif PVAR=='g':
    axs[0].set_title('G')
    lh, = axs[i].plot(xx, F(M,g,0,M).pdf(xx))

##

##
prnt = lambda s,v: print('%13.13s: %.5f'%(s,v))

##




