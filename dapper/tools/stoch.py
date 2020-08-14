# Random number generation

from dapper import *

def set_seed(i="clock",init=False):
    """Seed global random number generator by ``i``.

    Enables reproducibility of experiments which include randomness.

    If ``i==None``: the clock is used to seed.

    If ``init``: use 2 tricks to avoid accidentally re-using a seed:

      - Multiply i by 1000 (to leave room for 999 seeds for each i).
      - Add hostname_hash().

    Returns the seed, as a simplified state storage system.

    Note: it not possible to get the seed without setting it,
    because the mapping seed-->state is not surjective.

    Example
    >>> sd = set_seed(42); randn() # --> array([0.4967])
    >>> set_seed(sd);      randn() # --> array([0.4967])
    """
    if i in [None or "clock"]:
        np.random.seed()                # Init by clock.
        i = np.random.get_state()[1][0] # Set seed to state[0]

    if i==0:
        warnings.warn('''
        A seed of 0 is not a good idea. Use seed > 1.'
        [Sometimes people use sd_k = seed(k*sd_0) for experiment k.
        So if sd_0==0, then sd_k==0 for all k!]''')

    if init:
        i = i*1000 + hostname_hash()

    # Do the seeding
    np.random.seed(i)

    return i

import socket
def hostname_hash(max_hash=100000):
    """Compute an integer from hostname of computer.

    Used by set_seed() because it's difficult to know whether
    two computers will generate the same random numbers (given same seed).
    => Better to guarantee that they generate different random numbers.

    ``max_hash`` should be a large number so that different computers
    have hash values far apart and thus use entirely different ranges of seeds.
    """
    M = int(round(log10(max_hash)))          # Get max exponent
    h = socket.gethostname()                 # Get hostname str
    h = h*M                                  # ensure len(h)>=M
    h = h[:M]                                # make   len(h)==M
    chr_hasher = lambda a: np.mod(ord(a),10) # Define how one chr is hashed
    return sum(chr_hasher(a) * 10**expo for expo,a in enumerate(h))


def LCG(seed=-1):
    """(Unit) psuedo-random ("Linear congruential") generator.

    For X-platform use (since code is easy to translate).

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
