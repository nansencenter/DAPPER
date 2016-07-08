from common import *
    
def EnKF(params,cfg,xx,yy):
  """EnKF"""

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
    o_plt.update(E,k,kObs)
  return stats


def EnKF_analysis(E,hE,hnoise,y,cfg):
    if 'non-transposed' in cfg.AMethod:
      return EnKF_analysis_NT(E,hE,hnoise,y,cfg)

    R = hnoise.C
    N = cfg.N

    mu = mean(E,0)
    A  = E - mu

    hx = mean(hE,0)
    Y  = hE-hx
    dy = y - hx

    if 'PertObs' in cfg.AMethod:
      C  = Y.T @ Y + R.C*(N-1)
      D  = center(hnoise.sample(N))
      YC = mrdiv(Y, C)
      KG = A.T @ YC
      dE = (KG @ ( y + D - hE ).T).T
      #KG = mldiv(C,Y.T) @ A
      #dE = ( y + D - hE ) @ KG
      HK = Y.T @ YC
      E  = E + dE
    elif 'Sqrt' in cfg.AMethod:
      if 'explicit' in cfg.AMethod:
        # Implementation using inv (in ens space)
        Pw = inv(Y @ R.inv @ Y.T + (N-1)*eye(N))
        T  = sqrtm(Pw) * sqrt(N-1)
        KG = R.inv @ Y.T @ Pw @ A
        HK = R.inv @ Y.T @ Pw @ Y
      elif 'svd' in cfg.AMethod:
        # Implementation using svd of Y
        raise NotImplementedError
      else:
        # Implementation using eig. val.
        d,V= eigh(Y @ R.inv @ Y.T + (N-1)*eye(N))
        T  = V@diag(d**(-0.5))@V.T * sqrt(N-1)
        KG = R.inv @ Y.T @ (V@ diag(d**(-1)) @V.T) @ A
        HK = R.inv @ Y.T @ (V@ diag(d**(-1)) @V.T) @ Y
      if cfg.rot:
        T = genOG_1(N) @ T
      E = mu + dy@KG + T@A
    elif 'DEnKF' is cfg.AMethod:
      C  = Y.T @ Y + R.C*(N-1)
      KG = A.T @ mrdiv(Y, C)
      E  = E + KG@dy - 0.5*(KG@Y.T).T
    else:
      raise TypeError
    E = inflate_ens(E,cfg.infl)
    #if t<BurnIn:
      #E = inflate_ens(E,1.0 + 0.2*(BurnIn-t)/BurnIn)

    stat = {'trHK': trace(HK)/hnoise.m}
    return E, stat


def EnKF_analysis_NT(E,hE,hnoise,y,cfg):
    """Version: Non-Transposed. Purpose: debugging the other ones."""
    R = hnoise.C
    N = cfg.N

    E  = asmatrix(E).T
    hE = asmatrix(hE).T

    mu = mean(E,1)
    A  = E - mu
    hx = mean(hE,1)
    y  = y.reshape((hnoise.m,1))
    dy = y - hx
    Y  = hE-hx

    C  = Y@Y.T + R.C*(N-1)
    YC = mrdiv(Y.T, C)
    KG = A@YC
    HK = Y@YC
    D  = center(hnoise.sample(N)).T
    dE = KG @ ( y + D - hE )
    E  = E + dE
    E  = asarray(E.T)
    E  = inflate_ens(E,cfg.infl)

    stat = {'trHK': trace(HK)/hnoise.m}
    return E, stat





# TODO: MOD ERROR?
# TODO: It would be beneficial to do another (prior-regularized)
# analysis at the end, after forecasting the E0 analysis.
def iEnKF(params,cfg,xx,yy):
  """Loosely adapted from Bocquet ienks code and bocquet2014iterative."""
  f,h,chrono,X0,R = params.f, params.h, params.t, params.X0, params.h.noise.C
  N = cfg.N

  E = X0.sample(N)
  stats = Stats(params)
  stats.assess(E,xx,0)
  stats.iters = zeros(chrono.KObs+1)
  o_plt = LivePlot(params,E,stats,xx,yy)

  for kObs in progbar(range(chrono.KObs+1)):
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

    HK = R.inv @ Y.T @ Pw @ Y
    stats.trHK[kObs]  = trace(HK/h.noise.m)
    stats.iters[kObs] = iteration+1

    if cfg.rot:
      T = genOG_1(N) @ T
    T = T*cfg.infl

    E = xb0 + w @ A0 + T @ A0
    for k,t,dt in chrono.DAW_range(kObs):
      E  = f.model(E,t-dt,dt)
      E += sqrt(dt)*f.noise.sample(N)
      stats.assess(E,xx,k)
      #o_plt.update(E,k,kObs)
      
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



import numpy.ma as ma
def PartFilt(params,cfg,xx,yy):
  """
  Particle filter ≡ Sequential importance (re)sampling (SIS/SIR).
  This is the bootstrap version: the proposal density being just
  q(x_0:t|y_1:t) = p(x_0:t) = p(x_t|x_{t-1}) p(x_0:{t-1}).
  Resampling method: Multinomial.
  """

  f,h,chrono,X0 = params.f, params.h, params.t, params.X0

  N = cfg.N
  E = X0.sample(N)
  w = 1/N *ones(N)

  #Rm12 = inv(h.noise.C.cholL)
  Rm12 = h.noise.C.transform_by(lambda x: 1/np.sqrt(x))

  stats            = Stats(params)
  stats.N_eff      = zeros(chrono.KObs+1)
  stats.nResamples = 0
  stats.assess(E,xx,0)

  o_plt = LivePlot(params,E,stats,xx,yy)

  for k,kObs,t,dt in progbar(chrono.forecast_range):
    E  = f.model(E,t-dt,dt)
    E += sqrt(dt)*f.noise.sample(N)

    if kObs is not None:
      hE = h.model(E,t)
      y  = yy[kObs,:]
      innovs = hE - y
      innovs = innovs @ Rm12.T

      #naninds = np.isnan(np.sum(E,1))
      lklhds  = np.exp(-0.5 * np.sum(innovs**2, axis=1)) # *constant
      #lklhds[naninds] = 0
      lklhds = lklhds/mean(lklhds)/N # Avoid numerical error
      #%lklhds(lklhds==0) = min(lklhds(lklhds~=0))
      w *= lklhds
      #%w(w==0) = max(max(w)/N,1e-20)
      w /= sum(w)

      #log_lklhds = np.sum(innovs**2, axis=1) # +constant
      #w          = ma.masked_values(w,0)
      #log_w      = -2*log(w) + log_lklhds
      #log_w      = ma.masked_invalid(log_w)
      #log_w     -= log_w.mean() # avoid numerical error
      #w          = np.exp(-0.5*log_w)
      #w         /= np.sum(w)
      #w          = w.filled(0)

      N_eff = 1/(w@w)
      stats.N_eff[kObs] = N_eff
      # Resample
      if N_eff < N*cfg.NER:
        E = resample(E, w, N, f.noise)
        w = 1/N*ones(N)
        stats.nResamples += 1

    stats.assess_w(E,xx,k,w=w)
    o_plt.update(E,k,kObs)
  return stats


def resample(E,w,N,fnoise, \
    do_mu_corr=False,do_var_corr=False,kind='Multinomial'):
  """
  Resampling function for the particle filter.
  N can be different from E.shape[0] in case some particles
  have been elimintated.
  """
  N_b,m = E.shape

  # TODO A_b should be computed using weighted mean?
  # In DATUM, it wasn't.
  # Same question arises for Stats assessment.
  mu_b  = w@E
  A_b   = E - mu_b
  ss_b  = np.sqrt(w @ A_b**2)

  if kind is 'Multinomial':
    idx = np.random.choice(N_b,N,replace=True,p=w)
    E   = E[idx,:]

    if fnoise.is_deterministic:
      #If no forward noise: we need to add some.
      #Especially in the case of N >> m.
      #Use ss_b (which is precomputed, and weighted)?
      fudge = 4/sqrt(N)
      E += fudge * randn((N,m)) @ diag(ss_b)
  elif kind is 'Gaussian':
    N_eff = 1/(w@w)
    if N_eff<2:
      N_eff = 2
    ub = 1/(1 - 1/N_eff)
    A  = tp(sqrt(ub*w)) * A_b
    A  = randn((N,N)) @ A
    E  = mu_b + A
  else: raise TypeError

  # While multinomial sampling is unbiased, it does incur sampling error.
  # do_mu/var_corr compensates for this in the mean and variance.
  A_a,mu_a = anom(E)
  if do_mu_corr:
    # TODO: Debug
    mu_a = mu_b
  if do_var_corr:
    var_b = np.sum(ss_b**2)/m
    var_a = np.sum(A_a**2) /(N*m)
    A_a  *= np.sqrt(var_b/var_a)
  E = mu_a + A_a
    
  return E



def EnsCheat(params,cfg,xx,yy):
  """Ensemble method that cheats: it knows the truth.
  Nevertheless, its error will not be 0, because the truth may be outside of the ensemble subspace.
  This method is just to provide a baseline for comparison with other methods.
  It may very well beat the particle filter with N=infty.
  NB: The forecasts (and their rmse) are given by the standard EnKF.
  """

  f,h,chrono,X0 = params.f, params.h, params.t, params.X0

  E = X0.sample(cfg.N)

  stats = Stats(params)
  stats.assess(E,xx,0)
  o_plt = LivePlot(params,E,stats,xx,yy)

  for k,kObs,t,dt in progbar(chrono.forecast_range):
    E  = f.model(E,t-dt,dt)
    E += sqrt(dt)*f.noise.sample(cfg.N)

    if kObs is not None:
      # Regular EnKF analysis
      hE = h.model(E,t)
      y  = yy[kObs,:]
      E,s_now = EnKF_analysis(E,hE,h.noise,y,cfg)
      stats.copy_paste(s_now,kObs)

      # Cheating (only used for stats)
      w,res,_,_ = sla.lstsq(E.T, xx[k,:])
      if not res.size:
        res = 0
      res = sqrt(res/params.f.m) * ones(params.f.m)
      opt = w @ E

    stats.assess_ext(opt,res,xx,k)
    o_plt.update(E,k,kObs)
  return stats

def D3Var(params,cfg,xx,yy):
  """
  3D-Var.
  """
  #TODO: Take into account the dtObs/decorr time of the system,
  #and do the analysis with some smaller P than that of climatology.

  f,h,chrono,X0 = params.f, params.h, params.t, params.X0

  dkObs = chrono.dkObs
  R     = h.noise.C.C

  mu0   = np.mean(xx,0)
  A0    = xx - mu0
  P0    = (A0.T @ A0) / (xx.shape[0] - 1)

  mu = X0.mu

  # Pre-compute Kalman gain
  #dHdx = f.tlm
  #H    = dHdx(np.nan,mu0).T # TODO: .T ?
  H    = eye(h.m)
  KG   = mrdiv(P0 @ H.T, (H@P0@H.T) + R)
  Pa   = (eye(f.m) - KG@H) @ P0

  stats = Stats(params)
  stats.assess_ext(mu, sqrt(diag(P0)), xx, 0)
  stats.trHK[:] = trace(H@KG)/h.noise.m

  for k,kObs,t,dt in progbar(chrono.forecast_range):
    mu = f.model(mu,t-dt,dt)
    next_kobs = chrono.kkObs[find_1st_ind(chrono.kkObs >= k)]
    P  = Pa + (P0-Pa)*(1 - (next_kobs-k)/dkObs)
    if kObs is not None:
      y  = yy[kObs,:]
      mu = mu0 + KG @ (y - H@mu0)
    stats.assess_ext(mu,sqrt(diag(P)),xx,k)
  return stats


def ExtKF(params,cfg,xx,yy):
  pass






class Stats:
  """Contains and computes peformance stats."""
  # TODO: Include skew/kurt?
  def __init__(self,params):
    self.params = params
    m    = params.f.m
    K    = params.t.K
    KObs = params.t.KObs
    #
    self.mu   = zeros((K+1,m))
    self.var  = zeros((K+1,m))
    self.err  = zeros((K+1,m))
    self.rmsv = zeros(K+1)
    self.rmse = zeros(K+1)
    self.trHK = zeros(KObs+1)
    self.rh   = zeros((K+1,m))

  def assess(self,E,x,k):
    assert(type(E) is np.ndarray)
    N,m           = E.shape
    self.mu[k,:]  = mean(E,0)
    A             = E - self.mu[k,:]
    self.var[k,:] = np.sum(A**2,0) / (N-1)
    self.err[k,:] = self.mu[k,:] - x[k,:]
    self.rmsv[k]  = sqrt(mean(self.var[k,:]))
    self.rmse[k]  = sqrt(mean(self.err[k,:]**2))
    Ex_sorted     = np.sort(np.vstack((E,x[k,:])),axis=0,kind='heapsort')
    self.rh[k,:]  = [np.where(Ex_sorted[:,i] == x[k,i])[0][0] for i in range(m)]

  def assess_w(self,E,x,k,w):
    assert(type(E) is np.ndarray)
    assert(abs(sum(w)-1) < 1e-5)
    N,m           = E.shape
    self.mu[k,:]  = w @ E
    A             = E - self.mu[k,:]
    self.var[k,:] = w @ A**2
    self.err[k,:] = self.mu[k,:] - x[k,:]
    self.rmsv[k]  = sqrt(mean(self.var[k,:]))
    self.rmse[k]  = sqrt(mean(self.err[k,:]**2))

  def assess_ext(self,mu,ss,x,k):
    m             = len(mu)
    self.mu[k,:]  = mu
    self.var[k,:] = ss**2
    self.err[k,:] = self.mu[k,:] - x[k,:]
    self.rmsv[k]  = sqrt(mean(self.var[k,:]))
    self.rmse[k]  = sqrt(mean(self.err[k,:]**2))

  def copy_paste(self,s,kObs):
    for key,val in s.items():
      getattr(self,key)[kObs] = val

