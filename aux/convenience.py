from common import *

def simulate(setup,desc='Truth & Obs'):
  """Generate synthetic truth and observations."""
  f,h,chrono,X0 = setup.f, setup.h, setup.t, setup.X0

  # Init
  xx    = zeros((chrono.K+1,f.m))
  xx[0] = X0.sample(1)
  yy    = zeros((chrono.KObs+1,h.m))

  # Loop
  for k,kObs,t,dt in progbar(chrono.forecast_range,desc):
    xx[k] = f(xx[k-1],t-dt,dt) + sqrt(dt)*f.noise.sample(1)
    if kObs is not None:
      yy[kObs] = h(xx[k],t) + h.noise.sample(1)

  return xx,yy

def print_together(*args):
  "Print stacked 1D arrays."
  print(np.vstack(args).T)
