# Random number generation

from common import *


# (Unit) random number generator for x-platform use
# (since code is easy to translate).
# "Linear congruential generator"
# Should be burnt-in by running it a few times.
def LCG(seed=-1):
  if seed > -1:
    LCG.k = seed
  m = 2**32
  a = 1664525
  c = 1013904223
  LCG.k = (LCG.k * a + c) % m
  return float(LCG.k) / m
LCG.k = 1

# Unit pseudo-random numbers ucing LCG
def myrand(shape=(1,)):
  N = prod(shape)
  rand_U = [LCG() for k in range(N)]
  return reshape(rand_U, shape)

# Approximately Gaussian N(0,1),
# using logit transform of U(0,1) vars (from LCG)
def myrandn(shape=(1,)):
  u = myrand(shape)
  return sqrt(pi/8.) * log(u/(1-u))

# Built-in Gaussian generator
def randn(shape=(1,)): return np.random.normal(0,1,shape)
# Use LCG for all randn
#randn = myrandn


# Makeshift randcorr (which is missing from rogues)
def randcov(m):
  N = ceil(2+m**1.2)
  E = randn((N,m))
  return E.T @ E
def randcorr(m):
  Cov  = randcov(m)
  Dm12 = diag(diag(Cov)**(-0.5))
  return Dm12@Cov@Dm12


# Generate random orthonormal matrix:
def genOG(m):
  Q,R = nla.qr(randn((m,m)))
  for i in range(m):
    if R[i,i] < 0:
      Q[:,i] = -Q[:,i]
  return Q

# Random orthonormal mean-preserving matrix:
# Source: ienks code of Sakov/Bocquet
def genOG_1(N):
  e = ones((N,1))
  V = nla.svd(e)[0] # Basis whose first vector is e
  Q = genOG(N-1)     # Orthogonal mat
  return V @ sla.block_diag(1,Q) @ V.T

