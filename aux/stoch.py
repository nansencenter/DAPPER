# Random number generation

from common import *

#seed = np.random.seed
def seed(i):
  """Seed random number generator. Return input (for one-liners)."""
  np.random.seed(i)
  return i

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
# NB: To get equality with Datum, consider removing randomness altogether.
#     OR, to use LCG(), uncomment the above re-labellings,
#     and in Datum rename utils/myrand{,n}.m to utils/rand{,n}.m.
#     Also remember to reset seeds, and that in Datum:
#     - propagation (of truth) is fully completed before observation.
#     - observations are done from back to front.
#     - it seems that (somewhere?) a single draw is burnt
#       after x0 and before propagation (a little unsure...).
