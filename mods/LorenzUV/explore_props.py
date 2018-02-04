# Quick and dirty script
# to explore some properties of system

from common import *
seed(2)

# Note: seed(6) gives yields ICs that are in the basin for
# the tiny-variability limit cycle at $c=50$ (with Wilks setup),
# as can be seen by plotting up to T = 0.005 * 10**4.

###########################
# Setup
###########################
from mods.LorenzUV.wilks05 import LUV
LUV.F = 10

#from mods.LorenzUV.lorenz95 import LUV
#LUV.c = 1

nU = LUV.nU

K  = 2000
dt = 0.005
t0 = np.nan

x0 = 0.01*randn(LUV.m)

true_step  = with_rk4(LUV.dxdt      ,autonom=True)
model_step = with_rk4(LUV.dxdt_trunc,autonom=True)

###########################
# Compute truth trajectory
###########################
true_K = make_recursive(true_step,with_prog=1)
x0 = true_K(x0,int(2/dt),t0,dt)[-1] # BurnIn
xx = true_K(x0,K        ,t0,dt)

# ACF - make K to see periodicity
fig = plt.figure(6)
ax  = plt.gca()
ax.plot( mean( auto_cov(xx[:,:nU],L=K,corr=1), axis=1) )
plt.pause(0.1)

###########################
# Lyapunov estimation
###########################
if False:
  eps = 0.001
  U   = eye(LUV.m)
  E   = xx[0] + eps*U
  LL_exp = zeros((K,LUV.m))
  tt     = dt*(1+arange(K))
  for k,t in enumerate(progbar(tt,desc='Lyap')):
    E         = true_step(E,99,dt)
    E         = (E-xx[k+1]).T/eps
    [Q, R]    = sla.qr(E,mode='economic')
    E         = xx[k+1] + eps*Q.T
    LL_exp[k] = log(abs(diag(R)))

  # Running averages
  running_LS = ( tp(1/tt) * np.cumsum(LL_exp,axis=0) )
  LS         = running_LS[-1]
  n0 = sum(LS >= 0)
  print('c: ', LUV.c)
  print('var X: ', np.var(xx[:,:nU]))
  print('n0: ', n0)
  #
  plt.figure(7)
  plt.clf()
  plt.plot(tt,running_LS)

###########################
# Plot truth evolution
###########################
if True:
  plt.figure(8)
  ax = plt.gca()
  ax.clear()
  setter = LUV.plot_state(xx[0])
  for k in progbar(range(K),'plot'):
    if not k%(4*(int(0.005/dt))):
      setter(xx[k])
      ax.set_title("t = {:<5.2f}".format(dt*k))
      plt.pause(0.01)


###########################
# Properties as a func of:
# (setting: Wilks2005)
###########################
# h:   Var(xx[:,:nU])
# 0.1: 56.25
# 0.5: 43.91
# 1.0: 25.96
# 2.5: 0.123
# 3.0: 0.057

# c:    varX:  n0:  
# 1.00  54.62  59
# 10.0  25.35  58
# 20.0  10.44  33
# 30.0  3.157  32
# 40.0  0.409  19
# 50.0  0.103  7


