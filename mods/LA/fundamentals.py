import numpy as np
from scipy.linalg import circulant
from numpy import abs, sign, eye, ceil

def Fmat(m,c,dx,dt):
  # m  - System size
  # c  - Velocity of wave. Wave travels to the rigth for c>0.
  # dx - Grid spacing
  # dt - Time step
  #
  # CFL condition
  # Note that the 1st Ord Upwind scheme (i.e. F and dFdx) is exact
  # (vis-a-vis the analytic solution) for dt = abs(dx/c). 
  # In this case it corresponds to circshift. This has little bearing on
  # DA purposes, however.
  assert(abs(c*dt/dx) <= 1)
  # 1st order explicit upwind scheme
  row1     = np.zeros(m)
  row1[-1] = +(sign(c)+1)/2
  row1[+1] = -(sign(c)-1)/2
  row1[0]  = -1
  L        = circulant(row1)
  F        = eye(m) + (dt/dx*abs(c))*L;
  return asmatrix(F)

c    = -1;
dx   = 1;
F    = Fmod(m,c,dx,dt);
