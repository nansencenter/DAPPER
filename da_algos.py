from common import *
    

def EnKF(setup,config,xx,yy):
  """
  The EnKF.
  Settings for reproducing literature benchmarks may be found in
  mods/Lorenz95/sak08.py
  """

  f,h,chrono,X0,N = setup.f, setup.h, setup.t, setup.X0, config.N

  # Init
  E     = X0.sample(N)
  stats = Stats(setup,config).assess(E,xx,0)
  lplot = LivePlot(setup,config,E,stats,xx,yy)

  # Loop
  for k,kObs,t,dt in progbar(chrono.forecast_range):
    E = f.model(E,t-dt,dt)
    E = add_noise(E, dt, f.noise, config)

    if kObs is not None:
      hE = h.model(E,t)
      y  = yy[kObs]
      E  = EnKF_analysis(E,hE,h.noise,y,config.upd_a,stats.at(kObs))
      post_process(E,config)

    stats.assess(E,xx,k)
    lplot.update(E,k,kObs)
  return stats

def EnKF_NT(setup,config,xx,yy):
  """
  EnKF using 'non-transposed' analysis equations,
  where E is m-by-N, as is convention in EnKF litterature.
  This is slightly inefficient in our Python implementation,
  but is included for comparison (debugging, etc...).
  """

  f,h,chrono,X0,N = setup.f, setup.h, setup.t, setup.X0, config.N

  E     = X0.sample(N)
  stats = Stats(setup,config).assess(E,xx,0)
  lplot = LivePlot(setup,config,E,stats,xx,yy)

  for k,kObs,t,dt in progbar(chrono.forecast_range):
    E = f.model(E,t-dt,dt)
    E = add_noise(E, dt, f.noise, config)

    if kObs is not None:
      hE = h.model(E,t)
      y  = yy[kObs]

      E  = asmatrix(E).T
      hE = asmatrix(hE).T

      mu = mean(E,1)
      A  = E - mu
      hx = mean(hE,1)
      y  = y.reshape((h.m,1))
      dy = y - hx
      Y  = hE-hx

      C  = Y@Y.T + h.noise.C.C*(N-1)
      YC = mrdiv(Y.T, C)
      KG = A@YC
      HK = Y@YC
      D  = center(h.noise.sample(N)).T
      dE = KG @ ( y + D - hE )
      E  = E + dE
      E  = asarray(E.T)

      stats.trHK[kObs] = trace(HK)/h.m

      post_process(E,config)

    stats.assess(E,xx,k)
    lplot.update(E,k,kObs)
  return stats


def EnKS(setup,config,xx,yy):
  """
  EnKS.
  The only difference to the EnKF is the management of the lag and the reshapings.
  Settings for reproducing literature benchmarks may be found in
  mods/Lorenz95/sak08.py
  """

  f,h,chrono,X0,N = setup.f, setup.h, setup.t, setup.X0, config.N

  def reshape_to(E):
    K,N,m = E.shape
    return E.transpose([1,0,2]).reshape((N,K*m))
  def reshape_fr(E,m):
    N,Km = E.shape
    K    = Km/m
    return E.reshape((N,K,m)).transpose([1,0,2])

  E     = zeros((chrono.K+1,N,f.m))
  E[0]  = X0.sample(N)
  stats = Stats(setup,config)

  for k,kObs,t,dt in progbar(chrono.forecast_range):
    E[k] = f.model(E[k-1],t-dt,dt)
    E[k] = add_noise(E[k], dt, f.noise, config)

    if kObs is not None:
      kLag     = find_1st_ind(chrono.tt >= t-config.tLag)
      kkLag    = range(kLag, k+1)
      ELag     = E[kkLag]

      hE       = h.model(E[k],t)
      y        = yy[kObs]

      ELag     = reshape_to(ELag)
      ELag     = EnKF_analysis(ELag,hE,h.noise,y,config.upd_a,stats.at(kObs))
      E[kkLag] = reshape_fr(ELag,f.m)
      post_process(E[k],config)

  for k in progbar(range(chrono.K+1),desc='Assessing'):
    stats.assess(E[k],xx,k)

  return stats


def EnRTS(setup,config,xx,yy):
  """
  EnRTS (Rauch-Tung-Striebel) smoother.
  Settings for reproducing literature benchmarks may be found in
  mods/Lorenz95/sak08.py
  """

  f,h,chrono,X0,N = setup.f, setup.h, setup.t, setup.X0, config.N

  E     = zeros((chrono.K+1,N,f.m))
  Ef    = E.copy()
  E[0]  = X0.sample(N)
  stats = Stats(setup,config)

  # Forward pass
  for k,kObs,t,dt in progbar(chrono.forecast_range):
    E[k]  = f.model(E[k-1],t-dt,dt)
    E[k]  = add_noise(E[k], dt, f.noise, config)
    Ef[k] = E[k]

    if kObs is not None:
      hE   = h.model(E[k],t)
      y    = yy[kObs]
      E[k] = EnKF_analysis(E[k],hE,h.noise,y,config.upd_a,stats.at(kObs))
      post_process(E[k],config)

  # Backward pass
  for k in progbar(range(chrono.K)[::-1]):
    A  = anom(E[k])[0]
    Af = anom(Ef[k+1])[0]

    # TODO: make it tinv( , 0.99)
    J = sla.pinv2(Af) @ A
    J *= config.cntr
    
    E[k] += ( E[k+1] - Ef[k+1] ) @ J

  for k in progbar(range(chrono.K+1),desc='Assessing'):
    stats.assess(E[k],xx,k)
  return stats



def add_noise(E, dt, noise, config):
  """
  Treatment of additive noise for ensembles.
  Settings for reproducing literature benchmarks may be found in
  mods/LA/raanes2015.py
  """
  method = getattr(config,'fnoise_treatm','Stoch')

  N,m  = E.shape
  A,mu = anom(E)
  Q12  = noise.C.ssqrt
  Q    = noise.C.C

  def sqrt_core():
    T    = np.nan
    Qa12 = np.nan
    A2   = A.copy() # Instead of using: nonlocal A, which changes A
    # in outside scope as well. NB: This is a bug in Datum!
    if N<=m:
      Ainv = tinv(A2.T)
      Qa12 = Ainv@Q12
      T    = funm_psd(eye(N) + dt*(N-1)*(Qa12@Qa12.T), np.sqrt)
      A2   = T@A2
    else: # "Left-multiplying" form
      P = A2.T @ A2 /(N-1)
      L = funm_psd(eye(m) + dt*mrdiv(Q,P), np.sqrt)
      A2= A2 @ L.T
    E = mu + A2
    return E, T, Qa12

  if method == 'Stoch':
    # In-place addition works (also) for empty [] noise sample.
    E += sqrt(dt)*noise.sample(N)
  elif method == 'none':
    pass
  elif method == 'Mult-1':
    varE   = sum(np.var(E,axis=0,ddof=1))
    ratio  = (varE + sum(dt*diag(Q)))/varE
    E      = mu + sqrt(ratio)*A
    E      = reconst(*tsvd(E,0.999)) # Explained in Datum
  elif method == 'Mult-m':
    varE   = np.var(E,axis=0)
    ratios = sqrt( (varE + dt*diag(Q))/varE )
    E      = mu + A*ratios
    E      = reconst(*tsvd(E,0.999)) # Explained in Datum
  elif method == 'Sqrt-Core':
    E = sqrt_core()[0]
  elif method == 'Sqrt-Add-Z':
    E, _, Qa12 = sqrt_core()
    if N<=m:
      Z  = Q12 - A.T@Qa12
      E += sqrt(dt)*(Z@randn((Z.shape[1],N))).T
  elif method == 'Sqrt-Dep':
    E, T, Qa12 = sqrt_core()
    if N<=m:
      # Q_hat12: reuse svd for both inversion and projection.
      Q_hat12      = A.T @ Qa12
      U,s,VT       = tsvd(Q_hat12,0.99)
      Q_hat12_inv  = (VT.T * s**(-1.0)) @ U.T
      Q_hat12_proj = VT.T@VT
      # TODO: Make sqrt-core use chol instead of ssqrt factor of Q.
      # Then tsvd(Q_hat12) will be faster for LA where rQ=51.
      rQ = Q12.shape[1]
      # Calc D_til
      Z      = Q12 - Q_hat12
      D_hat  = A.T@(T-eye(N))
      Xi_hat = Q_hat12_inv @ D_hat
      Xi_til = (eye(rQ) - Q_hat12_proj)@randn((rQ,N))
      D_til  = Z@(Xi_hat + sqrt(dt)*Xi_til)
      E     += D_til.T
  else:
    raise KeyError('No such method')
  return E


def EnKF_analysis(E,hE,hnoise,y,upd_a,statFrame):
    R = hnoise.C
    N,m = E.shape

    mu = mean(E,0)
    A  = E - mu

    hx = mean(hE,0)
    Y  = hE-hx
    dy = y - hx

    if 'PertObs' in upd_a:
        # Uses perturbed observations (burgers'98)
        C  = Y.T @ Y + R.C*(N-1)
        D  = center(hnoise.sample(N))
        YC = mrdiv(Y, C)
        KG = A.T @ YC
        HK = Y.T @ YC
        dE = (KG @ ( y + D - hE ).T).T
        E  = E + dE
    elif 'Sqrt' in upd_a:
        # Uses a symmetric square root (ETKF)
        # to deterministically transform the ensemble.
        # The various versions below differ only numerically.
        if 'explicit' in upd_a:
          # Not recommended.
          # Implementation using inv (in ens space)
          Pw = inv(Y @ R.inv @ Y.T + (N-1)*eye(N))
          T  = sqrtm(Pw) * sqrt(N-1)
          HK = R.inv @ Y.T @ Pw @ Y
          #KG = R.inv @ Y.T @ Pw @ A
        elif 'svd' in upd_a:
          # Implementation using svd of Y R^{-1/2}.
          V,s,_ = svd0(Y @ R.m12.T)
          d     = pad0(s**2,N) + (N-1)
          Pw    = ( V * d**(-1.0) ) @ V.T
          T     = ( V * d**(-0.5) ) @ V.T * sqrt(N-1) 
          trHK  = sum( (s**2+(N-1))**(-1.0) * s**2 ) # see docs/trHK.jpg
        elif 'sS' in upd_a:
          # Same as 'svd', but with slightly different notation
          # (sometimes used by Sakov) using the normalization sqrt(N-1).
          #z    = dy@ R.m12.T / sqrt(N-1)
          S     = Y @ R.m12.T / sqrt(N-1)
          V,s,_ = svd0(S)
          d     = pad0(s**2,N) + 1
          Pw    = ( V * d**(-1.0) )@V.T / (N-1) # = G/(N-1)
          T     = ( V * d**(-0.5) )@V.T
          trHK  = sum(  (s**2 + 1)**(-1.0)*s**2 ) # see docs/trHK.jpg
        else: # 'eig' in upd_a:
          # Implementation using eig. val. decomp.
          d,V   = eigh(Y @ R.inv @ Y.T + (N-1)*eye(N))
          T     = V@diag(d**(-0.5))@V.T * sqrt(N-1)
          Pw    = V@diag(d**(-1.0))@V.T
          HK    = R.inv @ Y.T @ (V@ diag(d**(-1)) @V.T) @ Y
        w = dy @ R.inv @ Y.T @ Pw
        E = mu + w@A + T@A
    elif 'Serial' in upd_a:
        # Observations assimilated one-at-a-time.
        # Even though it's derived as "serial ETKF",
        # it's not equivalent to 'Sqrt' for the actual ensemble,
        # although it does yield the same mean/cov.
        # See DAPPER/Misc/batch_vs_serial.py for more details.
        inds = serial_inds(upd_a, y, R, A)
        z = dy@ R.m12.T / sqrt(N-1)
        S = Y @ R.m12.T / sqrt(N-1)
        T = eye(N)
        for j in inds:
          # Possibility: re-compute Sj by non-lin h.
          Sj = S[:,j]
          Dj = Sj@Sj + 1
          Tj = np.outer(Sj, Sj /  (Dj + sqrt(Dj)))
          T -= Tj @ T
          S -= Tj @ S
        GS   = S.T @ T
        E    = mu + z@GS@A + T@A
        trHK = trace(R.m12.T@GS@Y)/sqrt(N-1) # Correct?
    elif 'DEnKF' is upd_a:
        # Uses "Deterministic EnKF" (sakov'08)
        C  = Y.T @ Y + R.C*(N-1)
        YC = mrdiv(Y, C)
        KG = A.T @ YC
        HK = Y.T @ YC
        E  = E + KG@dy - 0.5*(KG@Y.T).T
    else:
        raise TypeError

    # Diagnostic: relative influence of observations
    if 'trHK' in locals():
      statFrame(trHK=trHK/hnoise.m)
    if 'HK'   in locals():
      statFrame(trHK=trace(HK)/hnoise.m)

    return E


def serial_inds(upd_a, y, cvR, A):
  if 'mono' in upd_a:
    # Not robust?
    inds = arange(len(y))
  elif 'sorted' in upd_a:
    dC = diag(cvR.C)
    if np.all(dC == dC[0]):
      # Sorty by P
      dC   = np.sum(A*A,0)/(N-1)
    inds = np.argsort(dC)
  else: # Default: random ordering
    inds = np.random.permutation(len(y))
  return inds
  

def SL_EAKF(setup,config,xx,yy):
  """
  Serial, covariance-localized EAKF.
  Used without localization, this should be equivalent
  (full ensemble equality) to the EnKF 'Serial'.
  See DAPPER/Misc/batch_vs_serial.py for some details.
  """
  f,h,chrono,X0,N = setup.f, setup.h, setup.t, setup.X0, config.N

  n = N-1

  R    = h.noise
  Rm12 = h.noise.C.m12
  #Ri   = h.noise.C.inv

  E     = X0.sample(N)
  stats = Stats(setup,config).assess(E,xx,0)
  lplot = LivePlot(setup,config,E,stats,xx,yy)

  for k,kObs,t,dt in progbar(chrono.forecast_range):
    E = f.model(E,t-dt,dt)
    E = add_noise(E, dt, f.noise, config)

    if kObs is not None:
      y    = yy[kObs]
      AMet = getattr(config,'upd_a','default')
      inds = serial_inds(AMet, y, R, anom(E)[0])
          
      for j in inds:
        hE = h.model(E,t)
        hx = mean(hE,0)
        Y  = (hE - hx).T
        mu = mean(E ,0)
        A  = E-mu

        Yj    = Rm12[j,:] @ Y
        dyj   = Rm12[j,:] @ (y - hx)

        skk   = Yj@Yj
        su    = 1/( 1/skk + 1/n )
        alpha = (n/(n+skk))**(0.5)

        dy2   = su*dyj/n # (mean is absorbed in dyj)
        Y2    = alpha*Yj

        # Localize
        local, coeffs = config.locf(j)
        if len(local) == 0: continue
        Regr  = (A[:,local]*coeffs).T @ Yj/sum(Yj**2)
        mu[ local] += Regr*dy2
        A[:,local] += np.outer(Y2 - Yj, Regr)

        # Without localization:
        #Regr = A.T @ Yj/sum(Yj**2)
        #mu  += Regr*dy2
        #A   += np.outer(Y2 - Yj, Regr)

        E = mu + A

      post_process(E,config)

    stats.assess(E,xx,k)
    lplot.update(E,k,kObs)
  return stats



def LETKF(setup,config,xx,yy):
  """
  Same as EnKF (sqrt), but with localization.
  """

  f,h,chrono,X0,N = setup.f, setup.h, setup.t, setup.X0, config.N

  Rm12 = h.noise.C.m12

  E     = X0.sample(N)
  stats = Stats(setup,config).assess(E,xx,0)
  lplot = LivePlot(setup,config,E,stats,xx,yy)

  for k,kObs,t,dt in progbar(chrono.forecast_range):
    E = f.model(E,t-dt,dt)
    E = add_noise(E, dt, f.noise, config)

    if kObs is not None:
      mu = mean(E,0)
      A  = E - mu

      hE = h.model(E,t)
      hx = mean(hE,0)
      YR = (hE-hx)  @ Rm12.T
      yR = (yy[kObs] - hx) @ Rm12.T

      for i in range(f.m):
        # Localize
        local, coeffs = config.locf(i)
        if len(local) == 0: continue
        iY  = YR[:,local] * sqrt(coeffs)
        idy = yR[local]   * sqrt(coeffs)

        # Do analysis
        upd_a = getattr(config,'upd_a','default')
        if upd_a is 'approx':
          # Approximate alternative, derived by pretending that Y_loc = H @ A_i,
          # even though the local cropping of Y happens after application of H.
          # Anyways, with an explicit H, one can apply Woodbury
          # to go to state space (dim==1), before reverting to HA_i = Y_loc.
          n   = N-1
          B   = A[:,i]@A[:,i] / n
          AY  = A[:,i]@iY
          BmR = AY@AY.T
          T2  = (1 + BmR/(B*n**2))**(-1)
          AT  = sqrt(T2) * A[:,i]
          P   = T2 * B
          dmu = P*(AY/(n*B))@idy
        elif upd_a is 'default':
          # Non-Approximate
          if len(local) < N:
            # SVD version
            V,sd,_ = svd0(iY)
            d      = pad0(sd**2,N) + (N-1)
            Pw     = (V * d**(-1.0)) @ V.T
            T      = (V * d**(-0.5)) @ V.T * sqrt(N-1)
          else:
            # EVD version
            d,V   = eigh(iY @ iY.T + (N-1)*eye(N))
            T     = V@diag(d**(-0.5))@V.T * sqrt(N-1)
            Pw    = V@diag(d**(-1.0))@V.T
          AT  = T@A[:,i]
          dmu = idy@iY.T@Pw@A[:,i]

        E[:,i] = mu[i] + dmu + AT

      post_process(E,config)

      if 'sd' in locals():
        stats.trHK[kObs] = sum(d**(-1.0) * sd**2)/h.noise.m
      #else:
        # nevermind

    stats.assess(E,xx,k)
    lplot.update(E,k,kObs)
  return stats


from scipy.optimize import minimize_scalar as minzs
def EnKF_N(setup,config,xx,yy):
  """
  Finite-size EnKF (EnKF-N).
  Corresponding to version ql2 of Datum.
  Settings for reproducing literature benchmarks may be found in
  mods/Lorenz95/sak08.py
  mods/Lorenz95/sak12.py
  """

  f,h,chrono,X0,N = setup.f, setup.h, setup.t, setup.X0, config.N

  # EnKF-N constants
  eN = (N+1)/N               # Effect of unknown mean
  g  = 1                     # Nullity of anomalies matrix # TODO: For N>m ?
  LB = sqrt((N-1)/(N+g)*eN)  # Lower bound for lambda^1    # TODO: Updated with g. Correct?
  clog = (N+g)/(N-1)         # Coeff in front of log term
  mode = eN/clog             # Mode of prior for lambda

  Rm12 = h.noise.C.m12
  Ri   = h.noise.C.inv

  E     = X0.sample(N)
  stats = Stats(setup,config).assess(E,xx,0)
  lplot = LivePlot(setup,config,E,stats,xx,yy)

  for k,kObs,t,dt in progbar(chrono.forecast_range):
    E = f.model(E,t-dt,dt)
    E = add_noise(E, dt, f.noise, config)

    if kObs is not None:
      hE = h.model(E,t)
      y  = yy[kObs]

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
      stats.at(kObs)(infl=l1)

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

      E = mu + w@A + T@A
      post_process(E,config)

      stats.trHK[kObs] = sum(((l1*s)**2 + (N-1))**(-1.0)*s**2)/h.noise.m

    stats.assess(E,xx,k)
    lplot.update(E,k,kObs)
  return stats



def iEnKF(setup,config,xx,yy):
  """
  Loosely adapted from Bocquet ienks code and bocquet2014iterative.
  Settings for reproducing literature benchmarks may be found in
  mods/Lorenz95/sak08.py
  mods/Lorenz95/sak12.py
  """
  f,h,chrono,X0,N,R = setup.f, setup.h, setup.t, setup.X0, config.N, setup.h.noise.C

  E     = X0.sample(N)
  stats = Stats(setup,config).assess(E,xx,0)
  lplot = LivePlot(setup,config,E,stats,xx,yy)

  for kObs in progbar(range(chrono.KObs+1)):
    xb0 = mean(E,0)
    A0  = E - xb0
    # Init
    w      = zeros(N)
    Tinv   = eye(N)
    T      = eye(N)
    for iteration in range(config.iMax):
      E = xb0 + w @ A0 + T @ A0
      for t,k,dt in chrono.DAW_range(kObs):
        E = f.model(E,t-dt,dt)
        E = add_noise(E, dt, f.noise, config)
  
      hE = h.model(E,t)
      hx = mean(hE,0)
      Y  = hE-hx
      Y  = Tinv @ Y
      y  = yy[kObs]
      dy = y - hx

      dw,Pw,T,Tinv = iEnKF_analysis(w,dy,Y,h.noise,config.upd_a)
      w  -= dw
      if np.linalg.norm(dw) < N*1e-4:
        break

    HK = R.inv @ Y.T @ Pw @ Y
    stats.trHK[kObs]  = trace(HK/h.noise.m)
    stats.at(kObs)(iters=iteration+1)

    E = xb0 + w @ A0 + T @ A0
    post_process(E,config)

    for k,t,dt in chrono.DAW_range(kObs):
      E = f.model(E,t-dt,dt)
      E = add_noise(E, dt, f.noise, config)
      stats.assess(E,xx,k)
      #lplot.update(E,k,kObs)
      
    # TODO: It would be beneficial to do another (prior-regularized)
    # analysis at the end, after forecasting the E0 analysis.
  return stats


def iEnKF_analysis(w,dy,Y,hnoise,upd_a):
  N = len(w)
  R = hnoise.C

  grad = (N-1)*w      - Y @ (R.inv @ dy)
  hess = (N-1)*eye(N) + Y @ R.inv @ Y.T

  if upd_a is 'PertObs':
    raise NotImplementedError
  elif 'Sqrt' in upd_a:
    if 'naive' in upd_a:
      Pw   = funm_psd(hess, np.reciprocal)
      T    = funm_psd(hess, lambda x: x**(-0.5)) * sqrt(N-1)
      Tinv = funm_psd(hess, np.sqrt) / sqrt(N-1)
    elif 'svd' in upd_a:
      # Implementation using svd of Y # TODO: sort out .T !!!
      raise NotImplementedError
    else:
      # Implementation using eig. val.
      d,V  = eigh(hess)
      Pw   = V@diag(d**(-1.0))@V.T
      T    = V@diag(d**(-0.5))@V.T * sqrt(N-1)
      Tinv = V@diag(d**(+0.5))@V.T / sqrt(N-1)
  elif upd_a is 'DEnKF':
    raise NotImplementedError
  else:
    raise NotImplementedError
  dw = Pw@grad

  return dw,Pw,T,Tinv



import numpy.ma as ma
def PartFilt(setup,config,xx,yy):
  """
  Particle filter ≡ Sequential importance (re)sampling (SIS/SIR).
  This is the bootstrap version: the proposal density being just
  q(x_0:t|y_1:t) = p(x_0:t) = p(x_t|x_{t-1}) p(x_0:{t-1}).
  Resampling method: Multinomial.
  """

  f,h,chrono,X0,N = setup.f, setup.h, setup.t, setup.X0, config.N

  Rm12 = h.noise.C.m12

  E = X0.sample(N)
  w = 1/N *ones(N)
  stats            = Stats(setup,config).assess(E,xx,0)
  stats.nResamples = 0

  lplot = LivePlot(setup,config,E,stats,xx,yy)

  for k,kObs,t,dt in progbar(chrono.forecast_range):
    E = f.model(E,t-dt,dt)
    E = add_noise(E, dt, f.noise, config)

    if kObs is not None:
      hE = h.model(E,t)
      y  = yy[kObs]
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
      stats.at(kObs)(N_eff=N_eff)
      # Resample
      if N_eff < N*config.NER:
        E = resample(E, w, N, f.noise)
        w = 1/N*ones(N)
        stats.nResamples += 1

    stats.assess(E,xx,k,w=w)
    lplot.update(E,k,kObs)
  return stats

def PF_EnKF(setup,config,xx,yy):
  """
  PF with EnKF proposal density: q.
  The setting infl_q tries to inflate the proposal density while
  keeping it centered on the EnKF posterior (by inflating both Q12 and R12).
  However, testing on L63, I can't get it to work as well as the standard PF,
  exceept with moderately small N, in which case the EnKF is already better.
  """
  f,h,chrono,X0,N = setup.f, setup.h, setup.t, setup.X0, config.N

  R     = h.noise.C.C
  Rm12T = h.noise.C.m12.T

  Q     = f.noise.C.C     * chrono.dtObs
  Qm12T = f.noise.C.m12.T / sqrt(chrono.dtObs)

  infl_q = config.infl_q

  E = X0.sample(N)
  w = 1/N *ones(N)

  stats            = Stats(setup,config).assess(E,xx,0)
  stats.nResamples = 0

  lplot = LivePlot(setup,config,E,stats,xx,yy)


  for k,kObs,t,dt in progbar(chrono.forecast_range):
    E  = f.model(E,t-dt,dt)

    if kObs is not None:
      E0 = E.copy()
      hE0= h.model(E0,t)
      DX = infl_q * sqrt(chrono.dtObs)*f.noise.sample(N)
      E += DX
      hE = h.model(E,t)
      y  = yy[kObs]

      mu = mean(E,0)
      A  = E - mu
      hx = mean(hE,0)
      Y  = hE-hx
      C  = Y.T @ Y + R*(N-1)
      YC = mldiv(C, Y.T)
      KG = YC @ A

      JH   = hjacob(mu,t)
      Sig2 = (eye(f.m) - KG.T*JH)*Q*(eye(f.m) - KG.T*JH).T + KG.T*R*KG
      Sig2 *= infl_q
      Sigm1= funm_psd(Sig2, lambda x: x**(-0.5))

      DY   = infl_q * h.noise.sample(N)
      E    = E + ( y + DY - hE )@KG
      post_process(E,config)

      # Sampling probabilities of the EnKF posterior ensemble.
      #qnd  = ( E - ( E0 + (y-hE0)@KG ) ) @ Sigm1
      qnd  = DX + ( DY - hE + hE0)@KG
      qnd  = qnd @ Sigm1


      # Transition probabilities of the EnKF posterior members
      # from the prior ensemble.
      pnd = (E-E0) @ Qm12T

      # Likelihood. NB: Must use posterior ensmeble
      lnd = (y - h.model(E,t)) @ Rm12T

      # New weights
      nrm2 = lambda w: np.sum(w**2, axis=1)
      w = -2*log(w) + nrm2(lnd) + nrm2(pnd) - nrm2(qnd)
      w = w - min(w) - 10 # Should calibrate via realmax/min
      w = np.exp(-0.5*w)
      w = w/np.sum(w)

      N_eff = 1/(w@w)
      stats.at(kObs)(N_eff=N_eff)
      # Resample
      if N_eff < N*config.NER:
        # NB: Must includte rescaling on f.noise if kind=Gaussian
        E = resample(E, w, N, f.noise, kind='Multinomial')
        w = 1/N*ones(N)
        stats.Neo = (getattr(stats,'Neo',0)*stats.nResamples + N_eff)/(stats.nResamples+1)
        stats.nResamples += 1

    stats.assess(E,xx,k,w=w)
    lplot.update(E,k,kObs)
  return stats



def GGM(setup,config,xx,yy):
  """
  "Global Gaussian Mixture".
  EnKF analysis proposal: q.
  But instead of weighting by ratio lklhd*(prior/q),
  it weights prior/prior, assuming lklhd taken care of.
  """


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
    E   = E[idx]

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



def EnCheat(setup,config,xx,yy):
  """
  Ensemble method that cheats: it knows the truth.
  Nevertheless, its error will not be 0, because the truth may be outside of the ensemble subspace.
  This method is just to provide a baseline for comparison with other methods.
  It may very well beat the particle filter with N=infty.
  NB: The forecasts (and their rmse) are given by the standard EnKF.
  """

  f,h,chrono,X0,N = setup.f, setup.h, setup.t, setup.X0, config.N

  E     = X0.sample(N)
  stats = Stats(setup,config).assess(E,xx,0)
  lplot = LivePlot(setup,config,E,stats,xx,yy)

  for k,kObs,t,dt in progbar(chrono.forecast_range):
    E = f.model(E,t-dt,dt)
    E = add_noise(E, dt, f.noise, config)

    if kObs is not None:
      # Regular EnKF analysis
      hE = h.model(E,t)
      y  = yy[kObs]
      E  = EnKF_analysis(E,hE,h.noise,y,config.upd_a,stats.at(kObs))
      post_process(E,config)

      # Cheating (only used for stats)
      w,res,_,_ = sla.lstsq(E.T, xx[k])
      if not res.size:
        res = 0
      res = sqrt(res/setup.f.m) * ones(setup.f.m)
      opt = w @ E
      # NB: It is also interesting to center E
      #     on the optimal solution.
      #E   = opt + E - mean(E,0)

    stats.assess_ext(opt,res,xx,k)
    lplot.update(E,k,kObs)
  return stats


def Climatology(setup,config,xx,yy):
  """
  A baseline/reference method.
  Note that the "climatology" is computed from truth,
  which might be (unfairly) advantageous if this simulation is too short.
  """
  f,h,chrono,X0 = setup.f, setup.h, setup.t, setup.X0

  mu0   = np.mean(xx,0)
  A0    = xx - mu0
  P0    = spCovMat(A=A0)

  stats = Stats(setup,config).assess_ext(mu0, sqrt(P0.diagonal), xx, 0)
  for k,_,_,_ in progbar(chrono.forecast_range):
    stats.assess_ext(mu0,sqrt(P0.diagonal),xx,k)
  return stats


def D3Var(setup,config,xx,yy):
  """
  3D-Var -- a baseline/reference method.
  A mix between Climatology() and the Extended KF.
  """
  f,h,chrono,X0 = setup.f, setup.h, setup.t, setup.X0

  dkObs = chrono.dkObs
  R     = h.noise.C.C
  #dHdx = fjacob
  #H    = dHdx(np.nan,mu0).T # TODO: .T ?
  H     = eye(h.m)

  mu0   = np.mean(xx,0)
  A0    = xx - mu0
  P0    = (A0.T @ A0) / (xx.shape[0] - 1)
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
  #config.infl = fsolve(L_minus_R, 0.9)
  if hasattr(config,'infl'):
    P0 *= config.infl
    
  # Pre-compute Kalman gain
  KG = mrdiv(P0 @ H.T, (H@P0@H.T) + R)
  Pa = (eye(f.m) - KG@H) @ P0

  mu = X0.mu

  stats = Stats(setup,config).assess_ext(mu, sqrt(diag(P0)), xx, 0)
  stats.trHK[:] = trace(H@KG)/h.noise.m

  for k,kObs,t,dt in progbar(chrono.forecast_range):
    mu = f.model(mu,t-dt,dt)
    next_kobs = chrono.kkObs[find_1st_ind(chrono.kkObs >= k)]
    P  = Pa + (P0-Pa)*(1 - (next_kobs-k)/dkObs)
    if kObs is not None:
      y  = yy[kObs]
      mu = mu0 + KG @ (y - H@mu0)
    stats.assess_ext(mu,sqrt(diag(P)),xx,k)
  return stats


def ExtKF(setup,config,xx,yy):
  f,h,chrono,X0 = setup.f, setup.h, setup.t, setup.X0

  R = h.noise.C.C
  Q = f.noise.C.C

  mu = X0.mu
  P  = X0.C.C

  stats = Stats(setup,config).assess_ext(mu, sqrt(diag(P)), xx, 0)

  for k,kObs,t,dt in progbar(chrono.forecast_range):
    
    F = fjacob(mu,t-dt,dt) 
    # "EKF for the mean". It's probably best to leave this commented
    # out because the benefit is negligable compared to the additional
    # cost incurred in estimating the Hessians.
    # HessCov = zeros(m,1) 
    # for k = 1:m
    #   HessianF_k = hessianest(@(x) submat(F(t,dt,x), k), X(:,iT-1))
    #   HessCov(k) = sum(sum( HessianF_k .* P(:,:,iT-1) ))
    # end
    # X(:,iT) = X(:,iT) + 1/2*HessCov 

    mu = f.model(mu,t-dt,dt)
    P  = F@P@F.T + dt*Q

    if kObs is not None:
      # NB: Inflation
      P *= config.infl

      H  = hjacob(mu,t)
      KG = mrdiv(P @ H.T, H@P@H.T + R)
      y  = yy[kObs]
      mu = mu + KG@(y - h.model(mu,t))
      KH = KG@H
      P  = (eye(f.m) - KH) @ P

      stats.trHK[kObs] = trace(KH)/f.m

    stats.assess_ext(mu,sqrt(diag(P)),xx,k)
  return stats




def post_process(E,config):
  """
  Inflate, Rotate.

  To avoid recomputing/recombining anomalies, this should be inside EnKF_analysis().
  But for readability it is nicer to keep it as a separate function,
  also since it avoids inflating/rotationg smoothed states (for the EnKS).
  """
  A, mu = anom(E)
  N,m   = E.shape
  T     = eye(N)
  try:
    T = config.infl * T
  except AttributeError:
    pass
  if getattr(config,'rot',False):
    T = genOG_1(N) @ T
  E[:] = mu + T@A




