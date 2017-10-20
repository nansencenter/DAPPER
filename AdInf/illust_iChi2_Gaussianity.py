# Show that the scaled inverse chi-square distribution
# converges to a (translating) Gaussian distribution
#
# UPDATE: I've now shown this analytically. See notes.

from common import *
from scipy.stats import norm, invgamma, chi2

# Convergence for t-->inf
t = 5

# Params
loc = 10*t
nu  = 50*t**2

# Manual iChi,Chi
def Chi2_pdf(s):
  c  = nu**(nu/2) / 2**(nu/2) / sp.special.gamma(nu/2)
  c /= loc**(nu/2)
  return c * s**(nu/2-1) * np.exp(-s*nu/2/loc)
def iChi2_pdf(s):
  c  = nu**(nu/2) / 2**(nu/2) / sp.special.gamma(nu/2)
  c *= loc**(nu/2)
  return c * s**(-nu/2-1) * np.exp(-nu*loc/2/s)
# Integrates to 1 ?
#from scipy.integrate import quad
#quad(iChi2_pdf,1e-10,100)

# From scipy
InvCS = lambda loc, nu: invgamma(a=nu/2,scale=loc*nu/2)
X = InvCS(loc,nu)

# Normal / Gaussian
N     = norm (loc,loc*sqrt(2/nu))
sig2  = (2*loc**2/nu)
N_pdf = lambda x: (2*pi*sig2)**(-1/2) * exp(-.5*(x-loc)**2/sig2)

###############
# Plotting
###############
#xx = np.linspace(10**-5,20,301)
xx = np.linspace(loc-3,loc+3,301)
fig, ax = plt.subplots()
ax.plot(xx,X.pdf    (xx),'b',label='InvCS',lw=3)
#ax.plot(xx,iChi2_pdf(xx),'g',label='InvCS Manual',ls='--') # Doesn't compute for large nu
ax.plot(xx,N.pdf    (xx),'r',label='N',lw=3)
ax.plot(xx,N_pdf    (xx),'y',label='N Manual',ls='--')
ax.set_title('loc= {:.1f}, nu = {:d}'.format(loc,nu))
plt.legend()

