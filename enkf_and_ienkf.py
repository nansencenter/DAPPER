from common import *


def EnKF_analysis(E,hE,hnoise,y,cfg):
    R = hnoise.C
    N = cfg.N

    mu = mean(E,0)
    A  = E - mu

    hx = mean(hE,0)
    Y  = hE-hx
    dy = y - hx

    if 'PertObs' in cfg.AMethod:
      if 'non-transposed' in cfg.AMethod:
        E  = asmatrix(E).T
        hE = asmatrix(hE).T

        mu = mean(E,1)
        A  = E - mu
        hx = mean(hE,1)
        y  = y.reshape((h.m,1))
        dy = y - hx
        Y  = hE-hx

        C  = Y@Y.T + R.C*(N-1)
        KG = A*mrdiv(Y.T, C)
        D  = center(hnoise.sample(N)).T
        dE = KG @ ( y + D - hE )
        dE = asarray(dE.T)
        E  = asarray(E.T)
      else:
        C  = Y.T @ Y + R.C*(N-1)
        D  = center(hnoise.sample(N))
        KG = A.T @ mrdiv(Y, C)
        dE = (KG @ ( y + D - hE ).T).T
        #KG = mldiv(C,Y.T) @ A
        #dE = ( y + D - hE ) @ KG
      E = E + dE
    elif 'Sqrt' in cfg.AMethod:
      if 'explicit' in cfg.AMethod:
        # Implementation using inv (in ens space)
        Pw = inv(Y @ R.inv @ Y.T + (N-1)*eye(N))
        T  = sqrtm(Pw) * sqrt(N-1)
        KG = R.inv @ Y.T @ Pw @ A
      elif 'svd' in cfg.AMethod:
        # Implementation using svd of Y
        raise NotImplementedError
      else:
        # Implementation using eig. val.
        d,V= eigh(Y @ R.inv @ Y.T + (N-1)*eye(N))
        T  = V@diag(d**(-0.5))@V.T * sqrt(N-1)
        KG = R.inv @ Y.T @ (V@ diag(d**(-1)) @V.T) @ A
      if cfg.rot:
        T = genOG_1(N) @ T
      E = mu + dy@KG + T@A
    elif 'DEnKF' is cfg.AMethod:
      C  = Y.T @ Y + R.C*(N-1)
      KG = A.T @ mrdiv(Y, C)
      E  = E + KG@dy - 0.5*(KG@Y.T).T
    else:
      raise ValueError
    E = inflate_ens(E,cfg.infl)
    #if t<BurnIn:
      #E = inflate_ens(E,1.0 + 0.2*(BurnIn-t)/BurnIn)

    stat = {'trK': trace(KG)}
    return E, stat


def EnKF(params,cfg,xx,yy):

  f,h,chrono,X0 = params.f, params.h, params.t, params.X0

  E = X0.sample(cfg.N)

  stats = Stats(params)
  stats.assess(E,xx,0)
  o_plt = LivePlot(params,E,stats,xx,yy)

  for k,kObs,t,dt in progbar(chrono.forecast_range):
    E  = f.model(E,t-dt,dt)
    E += sqrt(dt)*f.noise.sample(cfg.N)

    if kObs is not None:
      hE = h.model(E,t)
      y  = yy[kObs,:]
      E,s_now = EnKF_analysis(E,hE,h.noise,y,cfg)
      stats.copy_paste(s_now,kObs)

    stats.assess(E,xx,k)
    o_plt.update(E,k)
  return stats








def iEnKF_analysis(w,dy,Y,hnoise,cfg):
  N = len(w)
  R = hnoise.C

  grad = (N-1)*w      - Y @ (R.inv @ dy)
  hess = (N-1)*eye(N) + Y @ R.inv @ Y.T

  if cfg.AMethod is 'PertObs':
    raise NotImplementedError
  elif 'Sqrt' in cfg.AMethod:
    if 'naive' in cfg.AMethod:
      Pw   = funm_psd(hess, np.reciprocal)
      T    = funm_psd(hess, lambda x: x**(-0.5)) * sqrt(N-1)
      Tinv = funm_psd(hess, np.sqrt) / sqrt(N-1)
    elif 'svd' in cfg.AMethod:
      # Implementation using svd of Y
      raise NotImplementedError
    else:
      # Implementation using eig. val.
      d,V  = eigh(hess)
      Pw   = V@diag(d**(-1.0))@V.T
      T    = V@diag(d**(-0.5))@V.T * sqrt(N-1)
      Tinv = V@diag(d**(+0.5))@V.T / sqrt(N-1)
  elif cfg.AMethod is 'DEnKF':
    raise NotImplementedError
  else:
    raise NotImplementedError
  dw = Pw@grad

  return dw,Pw,T,Tinv


# Adapted from Bocquet ienks code and bocquet2014iterative
# TODO: MOD ERROR?
def iEnKF(params,cfg,xx,yy):
  f,h,chrono,X0,R = params.f, params.h, params.t, params.X0, params.h.noise.C
  N = cfg.N

  E = X0.sample(N)
  stats = Stats(params)
  stats.assess(E,xx,0)
  stats.iters = zeros(chrono.KObs)
  o_plt = LivePlot(params,E,stats,xx,yy)

  for kObs in progbar(range(chrono.KObs)):
    xb0 = mean(E,0)
    A0  = E - xb0
    # Init
    w      = zeros(N)
    Tinv   = eye(N)
    T      = eye(N)
    for iteration in range(cfg.iMax):
      E = xb0 + w @ A0 + T @ A0
      for t,k,dt in chrono.DAW_range(kObs):
        E  = f.model(E,t-dt,dt)
        E += sqrt(dt)*f.noise.sample(N)
  
      hE = h.model(E,t)
      hx = mean(hE,0)
      Y  = hE-hx
      Y  = Tinv @ Y
      y  = yy[kObs,:]
      dy = y - hx

      dw,Pw,T,Tinv = iEnKF_analysis(w,dy,Y,h.noise,cfg)
      w  -= dw
      if np.linalg.norm(dw) < N*1e-4:
        break

    stats.iters[kObs] = iteration+1
    stats.trK[kObs]   = trace(R.inv@(Y.T@(Pw@A0)))

    if cfg.rot:
      T = genOG_1(N) @ T
    T = T*cfg.infl

    E = xb0 + w @ A0 + T @ A0
    for k,t,dt in chrono.DAW_range(kObs):
      E  = f.model(E,t-dt,dt)
      E += sqrt(dt)*f.noise.sample(N)
      stats.assess(E,xx,k)
      #o_plt.update(E,k)
      

    # TODO: It would be beneficial to do another
    # (prior-regularized) analysis here, for the current ensemble

  return stats

