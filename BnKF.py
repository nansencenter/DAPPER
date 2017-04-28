# BnKF analysis step investigation
# Note: in contrast to GM, we here have:
# - unbiased variances (for GM: var = ens_var + kernel_var)
#   and the option to recreate the EnKF exactly (how?).
# - more reasonably-sized kernels
#   => +reasonable shifts and weightings
# - the freedom to choose the specific distribution
#   of the likelihood onto the different members.
# - weighting through the spread, instead of "importance".
#   the weighting will also scale more reasonably
#   (with thelikelihood being annealed with approx 1/N)
# - the weighting being through the spread, it will also affect
#   how the kernels are shifted.

# Directions of research:
# - different versions: equal mismatch, etc.
# - using a central carrier kernel (hybridizing with ETKF)
# - different solvers, initial guesses
# - use ETKF as usual, and reverse-estimate weights as in BnKF,
#   which last accross multiple cycles and hopefully improve
#   ensemble estimates withouth changing any members.
# - more general application (of mult mixture) than DA?

# TODO: Add Jacobian and do manual G-N ?
# TODO: Worry about root that corresponds to flipping the order of centres?
# TODO: Init guess based on DEnKF ?
# TODO: If root-finding fails, do ETKF

############################
# Preamble
############################
from common import *
from scipy.optimize import minimize_scalar as minz
import scipy.optimize as opt

seed(5)

try: del sol
except: pass

invm = lambda x: funm_psd(x, np.reciprocal)
inv2 = lambda x: funm_psd(x, lambda x: x**(-2.0))
m12  = lambda x: funm_psd(x, lambda x: x**(-1/2))


############################
# Set-up
############################
m  = 5
N  = 4
nu = N-1
cN = (N-1)/N

IN  = eye(N)
Pi1 = np.outer(ones(N),ones(N))/N
PiC = IN - Pi1

B     = randcov(m)
R     = randcov(m)
Ri    = invm(R)
P     = invm( Ri + invm(B) )

B_12  = funm_psd(B, sqrt)
R_12  = funm_psd(R, sqrt)


############################
# Sample ensemble
############################

E     = B_12 @ randn((m,N))
x     = B_12 @ randn(m)
y     = R_12 @ randn(m) + x

A, xb = anom(E,1)
Y, _  = anom(E,1)
Bb    = A@A.T / nu

# Prior kernel scalings
#bb = N*ones(N)
bb = N*rand(N)
bb*= np.sum(1/bb)


# Batch analysis (biased, coz no observation perturbations used)
# can be used for debugging.
# 1) Check that P2 == np.cov(E2,ddof=1)
# 2) Check that P2 == resulting_Pw(inpT(log(rr0))) [+ Pi1 which is "inconsequential"]
E2 = E + A@mrdiv(Y.T, Y@Y.T + R*(N-1)) @ ( tp(y) + 0 - E )
P2 = inv2( Y.T@Ri@Y/nu + eye(N) ) # If no ObsPert used.

# Prepare
Ea = np.full ((m,N), nan)
dC = np.zeros((N,N))
RY = m12(R) @ Y
_,s,VT = svd0(RY)


############################
# Do analysis
############################

# The target covariance is the "KF conforming" post cov.
target = invm( Y.T@Ri@Y/nu + eye(N) ) # = nu*Pwn

def resulting_Pw(rr):
  for n in arange(N):
    xn      = E[:,n]
    dn      = y-xn
    an      = nu*rr[n]/bb[n]
    #xn     = xn + A@Y.T @ invm( Y@Y.T + an*R ) @(y-xn+noise)
    #Pwn    = invm( RY.T @ RY/an + eye(N))/an
    dgn     = pad0(s**2,N) + an
    Pwn     = ( VT.T * dgn**(-1.0) ) @ VT
    Ea[:,n] = xn +    A@Pwn@Y.T@Ri@dn
    dC[:,n] = IN[:,n] + Pwn@Y.T@Ri@dn
  Ca = dC @ PiC @ dC.T
  return Ca


def inpT(logr):
  # Using exp ensures positivity.
  # Fixing last entry as N allows normalization.
  assert len(logr)==(N-1)
  rr = np.hstack([exp(logr), N])
  rr*= np.sum(1/rr)
  return rr


# 
rr0 = N*ones(N-1)
ops = {'maxiter':1000}


###########
def var_diff(logr):
  rr = inpT(logr)
  Ca = resulting_Pw(rr)
  diff = diag(PiC@(target-Ca)@PiC)
  return diff

def rootable(logr):
  return          var_diff(logr)[:-1]
def mismatch(logr):
  return nla.norm(var_diff(logr))**2
  
#sol = opt.root    (rootable, log(rr0), method='lm'  , options=ops)
#sol = opt.minimize(mismatch, log(rr0), method='BFGS', options=ops)

#print(inpT(sol.x))
#print(var_diff(sol.x))


###########
def r2(x):
  "Also optz. wrt. Pi1"
  rr = inpT(x[:-1])
  Ca = resulting_Pw(rr)
  diff = diag(  target - (Ca + x[-1]*Pi1)  )
  return diff

x0  = np.hstack([log(rr0), 1]) # augmented by coeff for Pi1
sol = opt.root(r2, x0, method='lm', options=ops)

print(inpT(sol.x[:-1]))
print(r2(sol.x))

# Compare target to:
# resulting_Pw( inpT(sol.x[:-1]) ) + sol.x[-1]*Pi1
# (should have equal diag but not necessarily off-diag)




