# Random number generation

from common import *

def LCG(seed=-1):
  """
  (Unit) random number generator for X-platform use
  (since code is easy to translate).
  "Linear congruential generator"
  Should be burnt-in by running it a few times.
  """  
  if seed > -1:
    LCG.k = seed
  m = 2**32
  a = 1664525
  c = 1013904223
  LCG.k = (LCG.k * a + c) % m
  return float(LCG.k) / m
LCG.k = 1

def myrand(shape=(1,)):
  """Unit pseudo-random numbers via LCG."""
  N = prod(shape)
  rand_U = [LCG() for k in range(N)]
  return reshape(rand_U, shape)

def myrandn(shape=(1,)):
  """
  Approximately Gaussian N(0,1).
  Using logit transform of U(0,1) vars (from LCG).
  """  
  u = myrand(shape)
  return sqrt(pi/8.) * log(u/(1-u))

# Built-in generators
def randn(shape=(1,)): return np.random.normal(0,1,shape)
def rand(shape=(1,)): return np.random.uniform(0,1,shape)

# Use LCG for all randn
#def randn(*kargs):
  #raise Warning('Using barebones LCG random numbers')
  #return myrandn(*kargs)
#def rand(*kargs):
  #raise Warning('Using barebones LCG random numbers')
  #return myrand(*kargs)
