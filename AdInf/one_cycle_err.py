# (Repeated) one-cycle experiments (i.e. no dynamics and no forecast/learning),
# with model error.

# To check consistency (CV of estimate to s2 as N-->infty) increase N,
# but beware that this quickly gets costly to simulate.

# For M11 (meth='Xplct R'):
#  - Expect(estimate_s2) = (s2+1/N)*nu/(nu-2) != s2 (i.e. it's biased),
#    where nu=N1*m. and HBH@inv(R) \propto eye(m) has been assumed. Thus,
#    - To observe bias due to division by B (i.e. trHPHR), use smallish m.
#    - To observe bias due to not accounting for the fact that dy is defined
#      through the **ensemble** mean, use small N.
# For meth='C' and meth='HBH' the bias is much worse
#   because the division happens before taking the trace.
#   Also the direction (pos/neg) of the bias is less clear.
# For meth 'VB heur I', use
#   - small m (eg m=2, N=100) to observe a severe bias (don't know why)
#   - large m (eg m=40, N=20) to also observe a severe bias
#   - m=5, N=50 to obsere a slight bias, but a better (smaller)
#     variance than 'Xplct R' [inspect with np.var(stat.b2_o)]


from common import *
from scipy.integrate import quad
from scipy.optimize import minimize_scalar as minz
from AdInf.filters import diag_HBH_I, diag_Gauss_logp, Chi2_logp
from scipy.optimize import fmin_bfgs, minimize_scalar


sd0 = seed()

meth = 'Xplct C'
K = 10**3
m = 20
N = 24
N1 = N-1
eN = (N+1)/N

# This is what infl^2 (here 'b2_o') should estimate
s2 = 0.2

B = eye(m) # randcov(m) diag(1+arange(m)**2) 
R = eye(m) # randcov(m) diag(1+arange(m)**2)
R = CovMat(R)
B = CovMat(B)

b = ones(m)
C = CovMat(R.full + B.full)


# Bunch of arrays for holding results
arr  = lambda K: np.full(K, nan)
stat = Bunch(b2_o=arr(K))
# For VB method
ITER  = 20
steps = zeros((K,ITER))
# afterwards, do: plot(abs(steps).mean(axis=0))

for k in progbar(range(K)):
  x  = b + randn(m)     @ B.Right
  hE = b + randn((N,m)) @ B.Right / sqrt(s2)
  y  = x + randn(m)     @ R.Right

  hx = mean(hE,0)
  Y  = hE-hx
  dy = y - hx

  dR    = dy @ R.sym_sqrt_inv.T
  YR    = Y  @ R.sym_sqrt_inv.T

  if 'Xplct' in meth:

    if 'R' in meth:
      trHPHR = trace(YR.T @ YR)/N1 # sum(s**2)/N1
      b2_o   = (dR@dR - m)/trHPHR
    elif 'HBH' in meth:
      V,s,UT = tsvd(YR)
      du     = UT @ dR
      trRHPH = sum(s**-2) * N1 # tr(R.full@tinv(Y.T@Y)*N1) = tr(tinv(YR.T@YR)*N1)
      b2_o   = ((N1*s**-2*du)@du - trRHPH)/len(s)
    elif 'C' in meth:
      C      = R.full + Y.T@Y/N1
      iC     = inv(C)
      b2_o   = trace((np.outer(dy,dy) - R.full)@iC) / trace((Y.T@Y/N1)@iC)

  elif 'ML' in meth:

    if 'Uni' in meth:
      # Solution available analytically, and equal to 'Xplct R' !
      trHPHR    = trace(YR.T @ YR)/N1 # sum(s**2)/N1
      log_lklhd = lambda b2: -Chi2_logp(m + trHPHR*b2, m, dR@dR)
      LB        = -0.99*m/trHPHR
    elif 'Mult' in meth:
      V,s,UT = svd0(YR) # could use tsvd for some methods?
      du     = UT @ dR

      dgn_v     = diag_HBH_I(s/sqrt(N1),min(N,m))
      log_lklhd = lambda b2: -diag_Gauss_logp(0, dgn_v(b2), du).sum(axis=1)
      LB        = -0.99*N1/max(s**2)
    b2_o = minimize_scalar(log_lklhd, bounds=(LB,9), method='Bounded').x

  elif 'VB heur' in meth:
    b2 = b2_o = 1 # init -- shouldn't matter coz using flat prior
    trHPH = trace(Y.T  @ Y )/N1 # sum(s**2)/N1
    if N<m: V,s,UT = svd0(YR)

    for itr in range(ITER):
      # Udate state with current estimate of inflation
      za = N1/b2
      if N<m:
        du     = UT @ dR
        Pw     = (V * (pad0(s**2,N) + za)**-1.0) @ V.T
        w      = dy @ R.inv @ Y.T @ Pw
        daf    = w@Y                 # innovation (analysis - forecast)
        trHPaH = trace(Y.T @ Pw @ Y) # trace analysis cov
      else:
        iC     = inv(YR.T@YR + za*eye(m))
        HPaH   = b2*Y.T@( eye(N) - YR@iC@YR.T )@Y
        trHPaH = trace(HPaH)/N1
        daf    = Y.T@YR@iC@dR
      
      # Update inflation with current estimate of state
      # Note: no point making 'R' method, coz Rm12 already applied.
      if 'I' in meth:
        b2 = (daf@daf + trHPaH)/trHPH
      elif 'af' in meth:
        b2 = daf@dy/trHPH

      steps[k,itr] = b2-b2_o
      b2_o = b2

  elif 'VB proper' in meth:
    b2 = b2_o = 1 # init -- shouldn't matter coz using flat prior
    V,s,UT = svd0(YR)
    for itr in range(ITER):
      za     = N1/b2
      B0     = Y.T @ Y / N1
      Ba     = Y.T @ Y / za
      iC     = inv(Ba + eye(m))
      Pa     = (eye(m) - Ba@iC) @ Ba
      dx     = Ba@iC@dR
      b2     = dx@inv(B0)@dx/m + trace(Pa @ inv(B0))/m

      # BUGGY. And: how to treat N<m case?
      #Pw     = (V * (pad0(s**2,N) + za)**-1.0) @ V.T
      #w      = dy @ R.inv @ Y.T @ Pw
      #b2     = za*w@w + trace(za*Pw)
      #b2    /= N

      steps[k,itr] = b2-b2_o
      b2_o = b2

  else:
    raise Exception("Method not defined")
  stat.b2_o[k] = b2_o



D = AlignedDict()
for key,val in stat.items():
  D["Empirical average of "+key] = str(series_mean_with_conf(val))
print(D)






