# Estimate the constant 1 by importance sampling
# Note the error in the estimate.

from common import *

#sd0 = seed(5)
N = 10**3

Nrml = lambda b,B: sp.stats.norm(loc=b, scale=sqrt(B))

# Prior moments
b = 0
B = 1
# Likelihood moments
R = 1
y = 0
# Bayes rule yields posterior moments:
P  = 1/(1/B+1/R)
mu = P*(b/B+y/R)


# Proposal: prior
q = Nrml(b, B)
# Target: posterior
p = Nrml(mu,P)
# Pdf ratio: Likelihood, normalized so that weights don't have to be self-normalized.
l = lambda x: p.pdf(x) / q.pdf(x)
#l = lambda x: Nrml(x,R).pdf(y) / Nrml(b,R+B).pdf(y) # Know by marginalization

# Sample ensemble
E = q.rvs(N)

statistic = lambda x: 1
#statistic = lambda x: x

estimate = 1/N * sum( statistic(E) * p.pdf(E) / q.pdf(E) )
