##

# Verify that importance sampling (IS) has bias.
# According to Doucet (2009):
# "A Tutorial on Particle Filtering and Smoothing: 15 years later"
# using NORMALIZED weights yields a BIAS (see below their eqn 26)
# in the PF (as defined by any test statistic).
# However, it remains consistent (CV to true dist for N-->infty).
# Also see owen2018importance below his eqn 9.3.

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt 
plt.ion()

#np.random.seed(2)

## True distribution
p = ss.uniform(loc=1)
#p = ss.norm(loc=1)

## Function for which we'll estimate the expected value
#f = lambda x: x**3             # nonlin f yields bias
f = lambda x: np.ones_like(x)  # constant f also yields bias

f_expect = f(p.rvs(10**4)).mean()

## Proposal
q = ss.norm()

# If proposal == true,
# then IS reduces to basic Monte-Carlo,
# and there should be no bias.
#q = p

xx = np.linspace(-2,8,401)
plt.plot(xx,p.pdf(xx),label='p')
plt.plot(xx,q.pdf(xx),label='q')
plt.legend()


## Bias estimation
K = 1000 # Experiment repetitions
N = 5    # Ensemble size

bias = []
for k in range(K):
  E = q.rvs(N)
  w = p.pdf(E)/q.pdf(E)

  # Note: Ignoring the case all(w==0) would introduce bias.
  #       In this case, just leave w==0.

  # Alternative 1: normalize weights.
  if not np.all(w==0):
    w /= w.sum()

  # Alternative 2: don't normalize. Should have no bias.
  #w /= N

  f_estim = w @ f(E)
  bias += [f_estim - f_expect]

print("Average err:",np.mean(bias))


##



