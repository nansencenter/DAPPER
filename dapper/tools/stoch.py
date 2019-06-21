# Random number generation

from dapper import *

def seed(i=None):
  """Seed global random number generator.

  If i==None, then the clock is used to seed.

  This (suboptimal) wrapper returns a seed,
  which can be used as a stored state (through re-seeding).
  It does not allow getting the state (as a seed), however,
  because the mapping seed-->state is not surjective.

  Example
  In : sd = seed(42); randn()
  Out: array([ 0.49671415])

  In : seed(sd); randn()
  Out: array([ 0.49671415])
  """
  if i==None:
    np.random.seed() # Init by clock.
    state = np.random.get_state()
    i     = state[1][0] # Set seed to state[0]
  if i==0:
    warnings.warn('''
    A seed of 0 is not a good idea. Use seed > 1.'
    [A convenient seed repetition system is to use
    seed_k = k*seed_0 for experiment k. But if seed_0 is 0,
    then all seed_k will be the same (0).]''')

  np.random.seed(i)
  return i

def seed_init(i=None):
  """Like seed(), but avoids accidentally re-using the same seed.

   What it does before calling seed():

    - multiply i by 1000 (to have 999 incremental seeds for each i)
    - and add hostname_hash() 
  """
  if i==None:
    i   = seed() # obtain clock-init
    sd0 = seed(     i + hostname_hash())
  else:
    sd0 = seed(i*1000 + hostname_hash())
  return sd0
  

import socket
def hostname_hash():
  """Generate integer from hostname.

  Add it to your initial seed to make this change
  as a function of your host computer.
  This guarantees different seeds are used for different computers.

  Reasoning:
  It's difficult to know whether different computers will generate
  the same random numbers. It is therefore better to guarantee that
  they generate different random numbers.
  """
  h = socket.gethostname()
  h = h[:2]+h[-2:]
  h = sum([10**q*np.mod(ord(c),10) for q,c in enumerate(h)])
  return h


def LCG(seed=-1):
  """(Unit) psuedo-random number generator for X-platform use (since code is easy to translate).

  "Linear congruential generator"
  Should be burnt-in by running it a few times.
  """  
  if seed > -1:
    LCG.k = seed
  M = 2**32
  a = 1664525
  c = 1013904223
  LCG.k = (LCG.k * a + c) % M
  return float(LCG.k) / M
LCG.k = 1

def myrand(shape=(1,)):
  """Unit pseudo-random numbers via LCG."""
  N = prod(shape)
  rand_U = [LCG() for k in range(N)]
  return reshape(rand_U, shape)

def myrandn(shape=(1,)):
  """Approximately Gaussian N(0,1).

  Using logit transform of U(0,1) vars (from LCG).
  """  
  u = myrand(shape)
  return sqrt(pi/8.) * log(u/(1-u))

# Use built-in generator
def rand( shape=(1,)): return np.random.uniform(0,1,shape)
def randn(shape=(1,)): return np.random.normal (0,1,shape)

