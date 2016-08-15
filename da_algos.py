from common import *
    
def EnKF(params,cfg,xx,yy):
  """EnKF"""

  f,h,chrono,X0 = params.f, params.h, params.t, params.X0

  E = X0.sample(cfg.N)

  stats = Stats(params,cfg)
  stats.assess(E,xx,0)
  lplot = LivePlot(params,E,stats,xx,yy)

  for k,kObs,t,dt in progbar(chrono.forecast_range):
    E  = f.model(E,t-dt,dt)
    E += sqrt(dt)*f.noise.sample(cfg.N)

    if kObs is not None:
      hE = h.model(E,t)
      y  = yy[kObs,:]
      E,s_now = EnKF_analysis(E,hE,h.noise,y,cfg)
      stats.copy_paste(s_now,kObs)

      # Spectral inflation
      if hasattr(cfg,'mv_infl'):
        A,mu = anom(E)
        V,s,UT = svd(A,full_matrices=False)
        s *= cfg.mv_infl
        A = (V*s) @ UT
        E = mu + A

    stats.assess(E,xx,k)
    lplot.update(E,k,kObs)
  return stats


def EnKS(params,cfg,xx,yy):
  """EnKS"""

  f,h,chrono,X0 = params.f, params.h, params.t, params.X0

  def reshape_to(E):
    K,N,m = E.shape
    return E.transpose([1,0,2]).reshape((N,K*m))
  def reshape_fr(E,m):
    N,Km = E.shape
    K = Km/m
    return E.reshape((N,K,m)).transpose([1,0,2])

  # Avoid inflating smoothed estimates
  # (even though a little would be good)
  cfg.infl, _infl = 1.0, cfg.infl
  # TODO: Should also avoid rotations

  E = zeros((chrono.K+1,cfg.N,f.m))
  E[0,:,:] = X0.sample(cfg.N)

  stats = Stats(params,cfg)

  for k,kObs,t,dt in progbar(chrono.forecast_range):
    E[k,:,:]  = f.model(E[k-1,:,:],t-dt,dt)
    E[k,:,:] += sqrt(dt)*f.noise.sample(cfg.N)

    if kObs is not None:
      kLag  = find_1st_ind(chrono.tt >= t-cfg.tLag)
      kkLag = range(kLag, k+1)
      ELag  = E[kkLag,:,:]

      hE = h.model(E[k,:,:],t)
      y  = yy[kObs,:]

      ELag = reshape_to(ELag)
      ELag,s_now = EnKF_analysis(ELag,hE,h.noise,y,cfg)
      stats.copy_paste(s_now,kObs)
      E[kkLag,:,:] = reshape_fr(ELag,f.m)
      E[k,:,:] = inflate_ens(E[k,:,:],_infl)

  for k in progbar(range(chrono.K+1),desc='Assessing'):
    stats.assess(E[k,:,:],xx,k)

  cfg.infl = _infl
  return stats


def EnRTS(params,cfg,xx,yy):
  """EnRTS (Rauch-Tung-Striebel)"""

  f,h,chrono,X0 = params.f, params.h, params.t, params.X0

  E  = zeros((chrono.K+1,cfg.N,f.m))
  Ef = E.copy()
  E[0,:,:] = X0.sample(cfg.N)

  stats = Stats(params,cfg)

  # Forward pass
  for k,kObs,t,dt in progbar(chrono.forecast_range):
    E[k,:,:]  = f.model(E[k-1,:,:],t-dt,dt)
    E[k,:,:] += sqrt(dt)*f.noise.sample(cfg.N)
    Ef[k,:,:] = E[k,:,:]

    if kObs is not None:
      hE = h.model(E[k,:,:],t)
      y  = yy[kObs,:]
      E[k,:,:],s_now = EnKF_analysis(E[k,:,:],hE,h.noise,y,cfg)
      stats.copy_paste(s_now,kObs)

  for k in progbar(range(chrono.K)[::-1],desc='Backward'):
    A  = anom(E[k,:,:])[0]
    Af = anom(Ef[k+1,:,:])[0]

    # TODO: make it tinv( , 0.99)
    J = sla.pinv2(Af) @ A
    J *= cfg.cntr
    
    E[k,:,:] += ( E[k+1,:,:] - Ef[k+1,:,:] ) @ J

  for k in progbar(range(chrono.K+1),desc='Assessing'):
    stats.assess(E[k,:,:],xx,k)
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
        # Uses perturbed observations (burgers'98)
        C  = Y.T @ Y + R.C*(N-1)
        D  = center(hnoise.sample(N))
        YC = mrdiv(Y, C)
        KG = A.T @ YC
        HK = Y.T @ YC
        dE = (KG @ ( y + D - hE ).T).T
        # Without transposing A:
        #KG = mldiv(C,Y.T) @ A
        #dE = ( y + D - hE ) @ KG
        E  = E + dE
    elif 'Sqrt' in cfg.AMethod:
        # Uses symmetric square root (ETKF)
        if 'explicit' in cfg.AMethod:
          # Implementation using inv (in ens space)
          Pw = inv(Y @ R.inv @ Y.T + (N-1)*eye(N))
          T  = sqrtm(Pw) * sqrt(N-1)
          HK = R.inv @ Y.T @ Pw @ Y
          #KG = R.inv @ Y.T @ Pw @ A
        elif 'svd' in cfg.AMethod:
          # Implementation using svd of Y R^{-1/2}
          V,s,_ = sla.svd( Y @ R.m12.T , full_matrices=False)
          d     = s**2 + (N-1)
          Pw    = ( V * d**(-1.0) ) @ V.T
          T     = ( V * d**(-0.5) ) @ V.T * sqrt(N-1) 
          trHK  = sum(  d**(-1.0)*s**2 ) # see docs/trHK.jpg
        elif 'sS' in cfg.AMethod:
          # Same as SVD, but with an initial sqrt(N-1) rescaling.
          s     = R.m12 @ dy  / sqrt(N-1)
          S     = Y @ R.m12.T / sqrt(N-1)
          V,s,_ = sla.svd(S, full_matrices=False)
          d     = s**2 + 1
          Pw    = ( V * d**(-1.0) )@V.T / (N-1) # = G/(N-1)
          T     = ( V * d**(-0.5) )@V.T
          trHK  = sum(  d**(-1.0)*s**2 )
        else:
          # Implementation using eig. val. decomp.
          d,V   = eigh(Y @ R.inv @ Y.T + (N-1)*eye(N))
          T     = V@diag(d**(-0.5))@V.T * sqrt(N-1)
          Pw    = V@diag(d**(-1.0))@V.T
          HK    = R.inv @ Y.T @ (V@ diag(d**(-1)) @V.T) @ Y
        if cfg.rot:
          T = genOG_1(N) @ T
        w = dy @ R.inv @ Y.T @ Pw
        E = mu + w@A + T@A
    elif 'DEnKF' is cfg.AMethod:
        # Uses "Deterministic EnKF" (sakov'08)
        C  = Y.T @ Y + R.C*(N-1)
        KG = A.T @ mrdiv(Y, C)
        E  = E + KG@dy - 0.5*(KG@Y.T).T
    else:
        raise TypeError
    E = inflate_ens(E,cfg.infl)
    #if t<BurnIn:
      #E = inflate_ens(E,1.0 + 0.2*(BurnIn-t)/BurnIn)

    # Diagnostic: relative influence of observations
    if 'HK' in locals():
      stat = {'trHK': trace(HK)/hnoise.m}
    else:
      stat = {'trHK': trHK/hnoise.m}

    return E, stat


def EnKF_analysis_NT(E,hE,hnoise,y,cfg):
    """
    Version: Non-Transposed.
    Purpose: debugging the other ones.
    """
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



from scipy.optimize import minimize_scalar as minzs

def EnKF_N(params,cfg,xx,yy):
  """
  Finite-size EnKF (EnKF-N).
  Corresponding to version ql2 of Datum.
  """

  f,h,chrono,X0 = params.f, params.h, params.t, params.X0

  N = cfg.N
  E = X0.sample(N)

  # EnKF-N constants
  eN = (N+1)/N;              # Effect of unknown mean
  g  = 1                     # Nullity of anomalies matrix # TODO: For N>m ?
  LB = sqrt((N-1)/(N+g)*eN); # Lower bound for lambda^1    # TODO: Updated with g. Correct?
  clog = (N+g)/(N-1);        # Coeff in front of log term
  mode = eN/clog;            # Mode of prior for lambda

  Rm12 = h.noise.C.m12
  Ri   = h.noise.C.inv

  stats = Stats(params,cfg)
  stats.assess(E,xx,0)
  stats.infl = zeros(chrono.KObs+1)
  lplot = LivePlot(params,E,stats,xx,yy)

  for k,kObs,t,dt in progbar(chrono.forecast_range):
    E  = f.model(E,t-dt,dt)
    E += sqrt(dt)*f.noise.sample(N)

    if kObs is not None:
      hE = h.model(E,t)
      y  = yy[kObs,:]

      mu = mean(E,0)
      A  = E - mu

      hx = mean(hE,0)
      Y  = hE-hx
      dy = y - hx

      V,s,U_T = sla.svd( Y @ Rm12.T )

      # Find inflation factor.
      du   = U_T @ (Rm12 @ dy)
      d    = lambda l: pad0( (l*s)**2, h.m ) + (N-1)
      PR   = sum(s**2)/(N-1)
      fctr = sqrt(mode**(1/(1+PR)))
      J    = lambda l: (du/d(l)) @ du \
             + (1/fctr)*eN/l**2 + fctr*clog*log(l**2)
      l1   = minzs(J, bounds=(LB, 1e2), method='bounded').x
      stats.infl[kObs] = l1

      # Turns it into the ETKF
      #l1 = 1.0

      # Inflate prior.
      # This is strictly equivalent to using zeta formulations.
      # With the Hessian adjustment, it's also equivalent to
      # the primal EnKF-N (in the Gaussian case).
      A *= l1
      Y *= l1

      # Compute ETKF (sym sqrt) update
      d       = lambda l: pad0( (l*s)**2, N ) + (N-1)
      Pw      = (V * d(l1)**(-1.0)) @ V.T
      w       = dy@Ri@Y.T@Pw
      T       = (V * d(l1)**(-0.5)) @ V.T * sqrt(N-1)

      # NB: Use Hessian adjustment ?
      # Replace sqrtm_psd with something like Woodbury?
      # zeta    = (N-1)/l1**2
      # Hess    = Y@Ri@Y.T + zeta*eye(N) \
      #           - 2*zeta**2/(N+g)*np.outer(w,w)
      # T       = funm_psd(Hess, lambda x: x**(-0.5)) * sqrt(N-1)

      if cfg.rot:
        T = genOG_1(N) @ T
      T *= cfg.infl

      E = mu + w@A + T@A

      stats.trHK[kObs] = sum(((l1*s)**2 + (N-1))**(-1.0)*s**2)/h.noise.m

    stats.assess(E,xx,k)
    lplot.update(E,k,kObs)
  return stats



# TODO: MOD ERROR?
# TODO: It would be beneficial to do another (prior-regularized)
# analysis at the end, after forecasting the E0 analysis.
def iEnKF(params,cfg,xx,yy):
  """Loosely adapted from Bocquet ienks code and bocquet2014iterative."""
  f,h,chrono,X0,R = params.f, params.h, params.t, params.X0, params.h.noise.C
  N = cfg.N

  E = X0.sample(N)
  stats = Stats(params,cfg)
  stats.assess(E,xx,0)
  stats.iters = zeros(chrono.KObs+1)
  lplot = LivePlot(params,E,stats,xx,yy)

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
      #lplot.update(E,k,kObs)
      
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
      # Implementation using svd of Y # TODO: sort out .T !!!
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

  Rm12 = h.noise.C.m12

  stats            = Stats(params,cfg)
  stats.N_eff      = zeros(chrono.KObs+1)
  stats.nResamples = 0
  stats.assess(E,xx,0)

  lplot = LivePlot(params,E,stats,xx,yy)

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
    lplot.update(E,k,kObs)
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

  stats = Stats(params,cfg)
  stats.assess(E,xx,0)
  lplot = LivePlot(params,E,stats,xx,yy)

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
      # NB: It is also interesting to center E
      #     on the optimal solution.
      #E   = opt + E - mean(E,0)

    stats.assess_ext(opt,res,xx,k)
    lplot.update(E,k,kObs)
  return stats


def Climatology(params,cfg,xx,yy):
  """
  A baseline/reference method.
  """
  f,h,chrono,X0 = params.f, params.h, params.t, params.X0

  mu0   = np.mean(xx,0)
  A0    = xx - mu0
  P0    = (A0.T @ A0) / (xx.shape[0] - 1)

  stats = Stats(params,cfg)
  stats.assess_ext(mu0, sqrt(diag(P0)), xx, 0)
  for k,_,_,_ in progbar(chrono.forecast_range):
    stats.assess_ext(mu0,sqrt(diag(P0)),xx,k)
  return stats

def D3Var(params,cfg,xx,yy):
  """
  3D-Var -- a baseline/reference method.
  A mix between Climatology() and the Extended KF.
  """
  f,h,chrono,X0 = params.f, params.h, params.t, params.X0

  dkObs = chrono.dkObs
  R     = h.noise.C.C
  #dHdx = f.tlm
  #H    = dHdx(np.nan,mu0).T # TODO: .T ?
  H    = eye(h.m)

  mu0   = np.mean(xx,0)
  A0    = xx - mu0
  P0    = (A0.T @ A0) / (xx.shape[0] - 1)
  if not hasattr(cfg, 'infl'):
    cfg.infl = 1.0
    ## NOT WORKING
    #from scipy.optimize import fsolve
    ##Take into account the dtObs/decorr_time of the system,
    ##by scaling P0 (from the climatology) by infl
    ##so that the error reduction of the analysis (trHK) approximately
    ##matches the error growth in the forecast (1-a^dkObs).
    #acovf = auto_cov(xx.ravel(order='F'))
    #a     = fit_acf_by_AR1(acovf)
    #def L_minus_R(infl):
      #KGs = mrdiv((infl*P0) @ H.T, (H@(infl*P0)@H.T) + R)
      #return trace(H@KGs)/h.noise.m - (1 - a**dkObs)
    #cfg.infl = fsolve(L_minus_R, 0.9)
  P0 *= cfg.infl
    
  # Pre-compute Kalman gain
  KG   = mrdiv(P0 @ H.T, (H@P0@H.T) + R)
  Pa   = (eye(f.m) - KG@H) @ P0

  mu = X0.mu

  stats = Stats(params,cfg)
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
  f,h,chrono,X0 = params.f, params.h, params.t, params.X0

  R = h.noise.C.C
  Q = f.noise.C.C

  mu = X0.mu
  P  = X0.C.C

  stats = Stats(params,cfg)
  stats.assess_ext(mu, sqrt(diag(P)), xx, 0)

  for k,kObs,t,dt in progbar(chrono.forecast_range):
    
    F = f.TLM(mu,t-dt,dt) 
    # "EKF for the mean". It's probably best to leave this commented
    # out because the benefit is negligable compared to the additional
    # cost incurred in estimating the Hessians.
    # HessCov = zeros(m,1);
    # for k = 1:m
    #   HessianF_k = hessianest(@(x) submat(F(t,dt,x), k), X(:,iT-1))
    #   HessCov(k) = sum(sum( HessianF_k .* P(:,:,iT-1) ))
    # end
    # X(:,iT) = X(:,iT) + 1/2*HessCov;

    mu = f.model(mu,t-dt,dt)
    P  = F@P@F.T + dt*Q

    if kObs is not None:
      # NB: Inflation
      P *= cfg.infl

      H  = h.TLM(mu,t)
      KG = mrdiv(P @ H.T, H@P@H.T + R)
      y  = yy[kObs,:]
      mu = mu + KG@(y - h.model(mu,t))
      KH = KG@H
      P  = (eye(f.m) - KH) @ P
      stats.trHK[kObs] = trace(KH)/f.m

    stats.assess_ext(mu,sqrt(diag(P)),xx,k)
  return stats






class Stats:
  """Contains and computes peformance stats."""
  # TODO: Include skew/kurt?
  def __init__(self,params,cfg):
    self.params = params
    m    = params.f.m
    K    = params.t.K
    KObs = params.t.KObs
    N    = cfg.N
    m_Nm = np.minimum(m,N)
    #
    self.mu    = zeros((K+1,m))
    self.var   = zeros((K+1,m))
    self.mad   = zeros((K+1,m))
    self.svals = zeros((K+1,m_Nm))
    self.umisf = zeros((K+1,m))
    self.smisf = zeros(K+1)
    self.ldet  = zeros(K+1)
    self.logp  = zeros(K+1)
    self.logp_r= zeros(K+1)
    self.logp_m= zeros(K+1)
    self.skew  = zeros(K+1)
    self.kurt  = zeros(K+1)
    self.err   = zeros((K+1,m))
    self.rmv  = zeros(K+1)
    self.rmse  = zeros(K+1)
    self.trHK  = zeros(KObs+1)
    self.rh    = zeros((K+1,m))

  def assess(self,E,x,k):
    assert(type(E) is np.ndarray)
    N,m             = E.shape
    self.mu[k,:]    = mean(E,0)
    A               = E - self.mu[k,:]
    self.var[k,:]   = sum(A**2  ,0) / (N-1)
    self.mad[k,:]   = sum(abs(A),0) / (N-1)
    self.skew[k]    = mean( sum(A**3,0)/N / self.var[k,:]**(3/2) )
    self.kurt[k]    = mean( sum(A**4,0)/N / self.var[k,:]**2 - 3 )
    self.err[k,:]   = x[k,:] - self.mu[k,:]
    self.rmv[k]    = sqrt(mean(self.var[k,:]))
    self.rmse[k]    = sqrt(mean(self.err[k,:]**2))

    # Marginal log score
    ldet            = sum(log(self.var[k]))
    nmisf           = self.var[k]**(-1/2) * self.err[k,:]
    logp_m          = sum(nmisf**2) + ldet
    self.logp_m[k]  = logp_m/m

    # Preparation for log score
    V,s,UT          = svd(A)
    s              /= sqrt(N-1)
    self.svals[k]   = s
    s               = s[s>1e-4]
    r               = np.minimum(len(s),5)
    s               = s[:r]

    # Full-joint Gaussian log score
    #alpha           = 1/20*mean(s)
    alpha           = 1e-2*sum(s)
    # Truncating s by alpha doesnt work:
    #s               = s[s>alpha]
    #r               = len(s)
    s2_full         = array(list(s**2) + [alpha]*(m-r))
    ldet            = sum(log(s2_full)) / m
    umisf           = UT @ self.err[k,:]
    nmisf           = (s2_full)**(-1/2) * umisf
    logp            = ldet + sum(nmisf**2)
    self.umisf[k,:] = umisf
    self.smisf[k]   = sum(nmisf**2)/m
    self.ldet[k]    = ldet/m
    self.logp[k]    = logp/m

    # Reduced-Joint Gaussian log score
    ldet            = sum(log(s**2))
    nmisf           = s**(-1) * (UT[:r,:] @ self.err[k,:])
    logp_r          = sum(nmisf**2) + ldet
    self.logp_r[k]  = logp_r/r

    # Rank histogram
    Ex_sorted       = np.sort(np.vstack((E,x[k,:])),axis=0,kind='heapsort')
    self.rh[k,:]    = [np.where(Ex_sorted[:,i] == x[k,i])[0][0] for i in range(m)]

  def assess_w(self,E,x,k,w):
    assert(type(E) is np.ndarray)
    assert(abs(sum(w)-1) < 1e-5)
    N,m           = E.shape
    self.mu[k,:]  = w @ E
    A             = E - self.mu[k,:]
    self.var[k,:] = w @ A**2
    self.err[k,:] = self.mu[k,:] - x[k,:]
    self.rmv[k]  = sqrt(mean(self.var[k,:]))
    self.rmse[k]  = sqrt(mean(self.err[k,:]**2))

  def assess_ext(self,mu,ss,x,k):
    m             = len(mu)
    self.mu[k,:]  = mu
    self.var[k,:] = ss**2
    self.err[k,:] = self.mu[k,:] - x[k,:]
    self.rmv[k]  = sqrt(mean(self.var[k,:]))
    self.rmse[k]  = sqrt(mean(self.err[k,:]**2))

  def copy_paste(self,s,kObs):
    """
    Load s into stats object at kObs.
    Avoids having to pass kObs into enkf_analysis (e.g.).
    """
    for key,val in s.items():
      getattr(self,key)[kObs] = val

