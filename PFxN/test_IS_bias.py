##
# Test if importance sampling has bias.
# According to Doucet (2009):
# "A Tutorial on Particle Filtering and Smoothing: 15 years later"
# using normalized weights YIELDS A BIAS (see below their eqn 26).


from common import *

#seed(2)

## True distribution

def p(xx):
  # Uniform(0,1)
  pp = np.ones_like(xx)
  pp[xx<0] = 0
  pp[xx>1] = 0
  return pp
p.sample = lambda N: rand(N)

#def p(xx):
  #return sp.stats.norm(loc=2).pdf(xx)
#p.sample = lambda N: 2 + randn(N)


## Function for which we'll estimate the expected value

#f = lambda x: x**3             # nonlin f yields bias
f = lambda x: x                # linear f yields bias
#f = lambda x: np.ones_like(x)  # constant f does not yield bias

f_expect = mean(f(p.sample(10**4)))


## Proposal

def q(xx):
  # Gaussian(0,1)
  return sp.stats.norm.pdf(xx)
q.sample = lambda N: randn(N)

# If using true distribution,
# then importance sampling reduces to basic Monte-Carlo,
# and there should be no bias.
#q = p

## 
#E = q.sample(10**5)
#Z = mean(p(E)/q(E))


## Bias estimation

K = 10000 # Experiment repetitions
N = 5     # Ensemble size

bias = []
for k in range(K):
  E = q.sample(N)
  w = p(E)/q(E)
  if np.all(w==0):
    continue
  #w /= w.sum()
  w /= N
  f_estim = w @ f(E)
  bias += [f_estim - f_expect]
bias = array(bias)

print("Average err:",bias.mean())


##



