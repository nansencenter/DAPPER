# Investigate whether the EnKF-N has a "one-cycle" intrinsic bias.
#
# TODO: Compute misfit as |x|_B/|x|_Bb
# TODO: Don't use iChi2 which is p(l2|E), 
#       but use pure likelihood for lambda (to get unbiased).
# TODO: Average over all innovations.

# TODO: What does this mean? In relation to the def of lambda?


from common import *
from scipy.integrate import quad
from scipy.optimize import minimize_scalar as minz

Domain = (1e-10, 200)
seed()

invm = lambda x: funm_psd(x, np.reciprocal)
m12  = lambda x: funm_psd(x, lambda x: x**(-1/2))

m  = 5
N  = 6 
eN = 1 + 1/N
nu = N-1
K  = 1000

arr  = lambda K: np.full(K, nan)
msft = Bunch(B=arr(K),P=arr(K),P_av=arr(K))

def iChi2_pdf(x2):
  c  = nu**(nu/2) / 2**(nu/2) / sp.special.gamma(nu/2)
  return c * x2**(-nu/2-1) * exp(-nu/2/x2)

def iChi2_gen(n):
  smpl = randn((nu, n))
  chi2 = np.sum(smpl**2,axis=0)/nu
  return 1/chi2

iChi2_mean = quad(lambda x: x*iChi2_pdf(x), *Domain)[0] # nu/(nu-2)
iChi2_mode = minz(lambda x:  -iChi2_pdf(x),  Domain).x  # nu/(nu+2)

# Debug. Compare:
# xx = linspace(*Domain,201)
# plt.plot(xx, iChi2_pdf(xx))
# plt.hist(iChi2_gen(100000),normed=True,bins=100,range=(0,5))

B     = randcov(m)
R     = randcov(m)
iR    = invm(R)
Y2S   = m12(nu*R)
P     = invm( iR + invm(B) )
B_12  = funm_psd(B, sqrt)

tR, tB, tP = trace(R), trace(B), trace(P)

for k in range(K):
  E     = B_12 @ randn((m,N))
  A, xb = anom(E,1)
  Bb    = A@A.T / nu
  iBb   = invm(Bb)
  Pb    = invm( iR + iBb)

  #Pb   = A @ invm(eye(N) + S.T@S) / nu @ A.T 
  #Pb   = A @ ( V * (pad0(s**2,N) + 1)**(-1.0) )@V.T / nu @ A.T
  S     = Y2S @ A
  V,s,_ = svd0(S.T)
  AV    = A@V
  d     = lambda l2: pad0(eN*l2*s**2,N) + 1
  trPb  = lambda l2: trace( (eN*l2)*(AV * d(l2)**(-1.0)) @ AV.T / nu)
  Pb_av = quad(lambda l2: iChi2_pdf(l2) * trPb(l2), *Domain)[0]

  msft.B[k]    = trace(Bb)/tB
  msft.P[k]    = trace(Pb)/tP
  msft.P_av[k] =     Pb_av/tP

  #l2s = iChi2_gen(K)
  #XX  = xb + eN*A/sqrt(N-1) @ randn((m,K)) * l2s
  

for key,val in msft.items():
  print("msft trace("+key.ljust(5)+") =",
      series_mean_with_conf(msft[key]))
