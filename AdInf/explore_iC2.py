# Investigate InvChi2Filter updating

from common import *
from AdInf.filters import *
from scipy.optimize import minimize_scalar as minz


########### Prior
self = InvChi2Filter(nu=50,sc=1)
DR   = 1e3

########### Likelihood
m      = 3
trHPHR = 3
dR2    = 6
log_lklhd = lambda b2: Chi2_logp(m + trHPHR*b2, m, dR2)


########### Posterior: log_update
for RUN in ['VALIDATION','ACTUAL']:
  if RUN is 'VALIDATION':
    # Make posterior==prior
    LL = lambda x: np.ones_like(x)
  else:
    LL = log_lklhd

  log_post  = lambda x: self.log_pdf(x) + LL(x)
  post_nn   = lambda x: exp(log_post(x) - log_post(self.sc))
  # Version with a single pdf evaluation
  xx        = linspace(*self.domain(DR), 1001)
  pp        = post_nn(xx)
  normlzt   = sum(pp)
  mean      = sum(          xx*pp)/normlzt
  var       = sum((xx-mean)**2*pp)/normlzt
  # Version using quadrature functions
  #_quad     = lambda f: sum(f(xx))*(xx[1]-xx[0]) # manual
  #_quad     = lambda f: quad(f,*self.domain(DR))[0]  # built-in
  #normlzt   = _quad(lambda x: post_nn(x))
  #post      = lambda x: post_nn(x) / normlzt
  #mean      = _quad(lambda x:  post(x)*x)
  #var       = _quad(lambda x:  post(x)*(x-mean)**2)
  if RUN is 'VALIDATION':
    # Estimate quadrature errors
    err_mean = mean - self.mean
    err_var  = var  - self.var
  else:
    # Update params
    mean      = mean - err_mean
    var       = var  - err_var
    #self.nu   = 4 + 2*mean**2/var
    #self.sc   = (self.nu-2)/self.nu * mean
    # NB Don't update in this script
    nu = 4 + 2*mean**2/var
    sc = (self.nu-2)/self.nu * mean


############
# Plotting
############
plt.figure(44).clear()
fig_ , (ax1,ax2) = plt.subplots(num=44,nrows=2,sharex=True)

# Normalization used: p(1) = 1 for all curves:
ax1.plot(xx, self.log_pdf(xx)  - self.log_pdf(1))
ax1.plot(xx, log_lklhd   (xx)  - log_lklhd   (1))
ax1.plot(xx, log(post_nn (xx)) - log(post_nn (1)),'-.')
#
ax2.plot(xx, exp(self.log_pdf(xx) - self.log_pdf(1) ) )
ax2.plot(xx, exp(log_lklhd   (xx) - log_lklhd   (1) ) )
ax2.plot(xx, post_nn(xx), '-.')

ax2.set_xlim(max(0,1-8/sqrt(self.nu)), 1+8/sqrt(self.nu))
ax2.hlines(0,*self.domain(DR))

############
# Compute new nu/sc
############
# Must come last to avoid it affecting the plots,
# because of late-binding in log_post
nu   = 4 + 2*mean**2/var
sc   = (self.nu-2)/self.nu * mean
print(nu,sc)


