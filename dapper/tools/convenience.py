from dapper import *

def simulate(HMM,desc='Truth & Obs'):
  """Generate synthetic truth and observations."""
  Dyn,Obs,chrono,X0 = HMM.Dyn, HMM.Obs, HMM.t, HMM.X0

  # Init
  xx    = zeros((chrono.K   +1,Dyn.M))
  yy    = zeros((chrono.KObs+1,Obs.M))

  xx[0] = X0.sample(1)

  # Loop
  for k,kObs,t,dt in progbar(chrono.ticker,desc):
    xx[k] = Dyn(xx[k-1],t-dt,dt) + sqrt(dt)*Dyn.noise.sample(1)
    if kObs is not None:
      yy[kObs] = Obs(xx[k],t) + Obs.noise.sample(1)

  return xx,yy


def rel_path(path,start=None,ext=False):
  path = os.path.relpath(path,start)
  if not ext:
    path = os.path.splitext(path)[0]
  return path

# TODO: archive
def simulate_or_load(script,HMM, sd, more): 
  t = HMM.t

  path = save_dir(rel_path(script)+'/sims/',pre=os.environ.get('SIM_STORAGE',''))

  path += 'HMM={:s} dt={:.3g} T={:.3g} dkObs={:d} {:s} Obs={:s} sd={:d}'.format(
      os.path.splitext(os.path.basename(HMM.name))[0],
      t.dt, t.T, t.dkObs,
      more,
      socket.gethostname(), sd)

  try:
    msg   = 'loaded from'
    data  = np.load(path+'.npz')
    xx,yy = data['xx'], data['yy']
  except FileNotFoundError:
    msg   = 'saved to'
    xx,yy = simulate(HMM)
    np.savez(path,xx=xx,yy=yy)
  print('Truth and obs',msg,'\n',path)
  return xx,yy
