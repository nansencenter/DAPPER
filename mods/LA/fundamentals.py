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
  F        = eye(m) + (dt/dx*abs(c))*L
  return F

def basis(m,k):
  """
  m - state vector length
  k - max wavenumber (wavelengths to fit into interval 1:m)
  """
  mm = arange(1,m+1) / m
  kk = np.arange(k+1) # Wavenumbers
  aa = rand(k+1)      # Amplitudes
  pp = rand(k+1)      # Phases

  s  = aa @ np.sin(2*pi*tp(kk) * (tp(pp) + mm))

  #% Normalise
  sd = np.std(s)
  #if m >= (2*k + 1)
      #% See analytic_normzt.m
      #sd = sqrt(sum(aa(2:end).^2)*(m/2)/(m-1));
  s  = s/sd

  return s

def X0pat(m,k,N):
  """ Generate N basis vectors """
  sample = zeros((N,m))
  for n in range(N):
    sample[n,:] = basis(m,k)

  #% Note: Each sample is centered -- Not the ensemble (in each dimension)
  sample = asmatrix(sample)
  sample = sample - np.mean(sample,1)
  sample = asarray(sample)
  return sample 

  
