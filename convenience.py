from common import *

def assimilate(setup,config,xx,yy):
  """Call config.da_driver(), passing along all arguments."""
  args = locals()
  return config.da_driver(**args)
  # try:
  #   stats = config.da_driver(**args)
  # except Exception as e:
  #   tb = e.__traceback__
  #   #warnings.warn() Doesn't print color. 
  #   print_c('''Assimilation failed. Traceback:''',color='FAIL')
  #   traceback.print_tb(tb)
  #   stats = Stats(setup,config)
  #   stats.has_failed = True
  #   print_c("Returning empty stats object, " +
  #       "and resuming execution.",color='WARNING')
  # return stats

def simulate(setup):
  """Generate synthetic truth and observations"""
  f,h,chrono,X0 = setup.f, setup.h, setup.t, setup.X0

  # init
  xx    = zeros((chrono.K+1,f.m))
  xx[0] = X0.sample(1)
  yy    = zeros((chrono.KObs+1,h.m))

  for k,kObs,t,dt in progbar(chrono.forecast_range,desc='Truth & Obs'):
    xx[k] = f.model(xx[k-1],t-dt,dt) + sqrt(dt)*f.noise.sample(1)
    if kObs is not None:
      yy[kObs] = h.model(xx[k],t) + h.noise.sample(1)

  return xx,yy

