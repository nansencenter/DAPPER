# Quick and dirty script
# to explore some properties of system

from common import *
seed(2)

###########################
# Setup
###########################
import mods.LorenzXY.core as LXY
LXY.c = 0.01
from mods.LorenzXY.core import *
from mods.LorenzXY.defaults import plot_state

K  = 2000
dt = 0.005
t0 = np.nan

x0 = 0.01*randn(m)

true_step  = with_rk4(dxdt      ,autonom=True)
model_step = with_rk4(dxdt_trunc,autonom=True)

###########################
# Compute truth trajectory
###########################
true_K = make_recursive(true_step,with_prog=1)
x0 = true_K(x0,int(2/dt),t0,dt)[-1] # BurnIn
xx = true_K(x0,K        ,t0,dt)

# ACF
#plt.plot( mean( auto_cov(xx[:,:nX],L=K,corr=1), axis=1) )

###########################
# Lyapunov estimation
###########################
if False:
  eps = 0.001
  U   = eye(LXY.m)
  E   = xx[0] + eps*U
  LL_exp = zeros((K,LXY.m))
  tt     = dt*(1+arange(K))
  for k,t in progbar(enumerate(tt),desc='Lyap'):
    E         = true_step(E,99,dt)
    E         = (E-xx[k+1]).T/eps
    [Q, R]    = sla.qr(E,mode='economic')
    E         = xx[k+1] + eps*Q.T
    LL_exp[k] = log(abs(diag(R)))

  # Running averages
  running_LS = ( tp(1/tt) * np.cumsum(LL_exp,axis=0) )
  LS         = running_LS[-1]
  n0 = sum(LS >= 0)
  print('c: ', LXY.c)
  print('var X: ', np.var(xx[:,:nX]))
  print('n0: ', n0)
  #
  plt.figure(1)
  plt.clf()
  plt.plot(tt,running_LS)

###########################
# Plot truth evolution
###########################
if True:
  plt.figure(2)
  setter = plot_state(xx[0])
  ax = plt.gca()
  for k in progbar(range(min(400,K)),'plot'):
    if not k%4:
      setter(xx[k])
      ax.set_title("t = {:<5.2f}".format(dt*k))
      plt.pause(0.01)


# h:   Var(xx[:,:nX])
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


