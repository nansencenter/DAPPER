
# Lorenz95 is highly sensitive to large gradients.
# Therefore, if we only observe every 4th (e.g.) state component,
# the members might "blow up" during the forecast,
# because the assimilation created large gradients.
# (Of course, this will depend on R and dtObs).
# Therefore, the setup below instead uses "global obs",
# where each observation "releve"/captures information about
# the entire state vector.
# The idea is that we can then remove observations, (rows of H)
# one-by-one, to a much larger degree than for H = Identity.

# Ideally, we want the observations to be independent,
# and possibly of the same magnitude
# (i.e. we that the rows of H be orthonormal).
# Furthermore, we want that each observation gives equal weight
# ("consideration") to each state component.
# This can be shown to be equivalent to requiring that the state
# component is equally resolved by each observation,
# and moreover, that the magnitude (abs) of each element of H
# be a constant (1/sqrt(m)).

# Can such an H be constructed/found?
# In the 2d case: H = [1, 1; 1, -1] / sqrt(2).
# In the 3d case: no, as can be shown by enumeration.
# (note, however, how easy my geometric intuition was fooled.
# Try rotating the 3-dim stensil.
# Intuitively I thought that it would yield the 3d H of +/- 1's).
# ...
# In fact, only in the 2^n - dimensional case is it possible
# (our conjecture: Madleine/Patrick, based on analogy with the FFT).
#
# Another idea is then to evaluate the value of 40 orthogonal basis
# functions at 40 equidistant locations
# (corresponding to the indices of Lorenz-96).
# This will not yield a matrix of +/- 1's,
# but should nevertheless give nicely distributed weights.

# Note that the legendre polynomials are not orthogonal when
# (the inner product is) evaluated on discrete, equidistant points.
# Moreover, the actual orthogonal polynomial basis
# (which I think goes under the name of Gram polynomials, and can be
# constructed by qr-decomp (gram-schmidt) of a 40-dim Vandermonde matrix).
# would in fact not be a good idea:
# it is well-known that not only will the qr-decomp be numerically unstable,
# the exact polynomial interpolant of 40 equidistant points
# is subject to the "horrible" Runge phenomenon.
# 
# Another basis is the harmonic (sine/cosine) functions. Advantages:
#  - will be orthogonal when evaluated on 40 discrete equidistant points.
#  - the domain of Lorenz95 is periodic: as are sine/cosine.
#  - (conjecture) in the 2^n dim. case, it yields a matrix of +/- 1's.
#  - nice "spectral/frequency" interpretation of each observation.
# Disadvatages:
#  - 40 is not 2^n for any n
#  - Too obvious (not very original).
#
# In conclusion, we will use the harmonic functions.
#
# 
# Update: It appears that we were wrong concerning the 2^n case.
# That is, sine/cosine functions do not yield only +/- 1's for n>2
# The question then remains (to be proven combinatorically?)
# if the +/- 1 matrices exist for dim>4
# Furthermore, experiments do not seem to indicate that I can push
# p much lower than for the case H = Identity,
# even though the rmse is a lot lower with spectral H.
# Am I missing something?


from mods.Lorenz95.sak08 import *

# The (m-p) highest frequency observation modes are
# left out of H below.
# If p>m, then H no longer has independent
# (let alone orthogonal) columns,
# yet more information is gained, since the
# observations are noisy.
p = 12


mu0,P0 = typical_init_params(m)
X0 = GaussRV(mu0, 0.01*P0)

def make_H(p,m):
  xx = linspace(-1,1,m+1)[1:]
  H = zeros((p,m))
  H[0] = 1/sqrt(2)
  for k in range(-(p//2),(p+1)//2):
    ind = 2*abs(k) - (k<0)
    H[ind] = sin(pi*k*xx + pi/4)
  H /= sqrt(m/2)
  return H

H = make_H(p,m)
#plt.figure(1).gca().matshow(H)
#plt.figure(2).gca().matshow(H @ H.T)


# "raw" obs plotting
#if p<=m:
  #Hinv = H.T
#else:
  #Hinv = sla.pinv2(H)
#def yplot(y):
  #lh = plt.plot(y @ Hinv.T,'g')[0]
  #return lh

# "implicit" (interpolated sine/cosine) obs plotting
Hplot = make_H(p,max(p,201))
Hplot_inv = Hplot.T
def yplot(y):
  x = y @ Hplot_inv.T
  ii = linspace(0,m-1,len(x))
  lh = plt.plot(ii,x,'g')[0]
  return lh

h = {
    'm': p,
    'model': lambda x,t: x @ H.T,
    'noise': GaussRV(C=0.01*eye(p)),
    'plot' : yplot,
    }

other = {'name': os.path.relpath(__file__,'mods/')}

setup = OSSE(f,h,t,X0,**other)


####################
# Suggested tuning
####################
#config = EnKF ('Sqrt',N=40, infl=1.01)

