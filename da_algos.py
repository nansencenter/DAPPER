from common import *
    

def EnKF(setup,config,xx,yy):
  """
  The EnKF.

  Ref: Evensen, Geir. (2009):
  "The ensemble Kalman filter for combined state and parameter estimation."

  Settings for reproducing literature benchmarks may be found in
  mods/Lorenz95/sak08.py
  """

  # Unpack
  f,h,chrono,X0 = setup.f, setup.h, setup.t, setup.X0
  N, upd_a      = config.N, config.upd_a

  # Init
  E     = X0.sample(N)
  stats = Stats(setup,config,xx,yy).assess(0,E=E)

  # Loop
  for k,kObs,t,dt in progbar(chrono.forecast_range):
    E = f.model(E,t-dt,dt)
    E = add_noise(E, dt, f.noise, config)

    # Analysis update
    if kObs is not None:
      stats.assess(k,kObs,'f',E=E)
      hE = h.model(E,t)
      y  = yy[kObs]
      E  = EnKF_analysis(E,hE,h.noise,y,upd_a,stats,kObs)
      post_process(E,config)

    stats.assess(k,kObs,E=E)

  return stats

def EnKF_tp(setup,config,xx,yy):
  """
  EnKF using 'non-transposed' analysis equations,
  where E is m-by-N, as is convention in EnKF litterature.
  This is slightly inefficient in our Python implementation,
  but is included for comparison (debugging, etc...).
  """

  f,h,chrono,X0,N = setup.f, setup.h, setup.t, setup.X0, config.N

  E     = X0.sample(N)
  stats = Stats(setup,config,xx,yy).assess(0,E=E)

  for k,kObs,t,dt in progbar(chrono.forecast_range):
    E = f.model(E,t-dt,dt)
    E = add_noise(E, dt, f.noise, config)

    if kObs is not None:
      stats.assess(k,kObs,'f',E=E)
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

    stats.assess(k,kObs,E=E)
  return stats


def EnKS(setup,config,xx,yy):
  """
  EnKS (ensemble Kalman smoother)

  Ref: Evensen, Geir. (2009):
  "The ensemble Kalman filter for combined state and parameter estimation."

  The only difference to the EnKF is the management of the lag and the reshapings.
  Settings for reproducing literature benchmarks may be found in
  mods/Lorenz95/raanes2016.py
  """

  f,h,chrono,X0 = setup.f, setup.h, setup.t, setup.X0
  N, upd_a      = config.N, config.upd_a

  def reshape_to(E):
    K,N,m = E.shape
    return E.transpose([1,0,2]).reshape((N,K*m))
  def reshape_fr(E,m):
    N,Km = E.shape
    K    = Km//m
    return E.reshape((N,K,m)).transpose([1,0,2])

  E     = zeros((chrono.K+1,N,f.m))
  E[0]  = X0.sample(N)
  stats = Stats(setup,config,xx,yy)

  for k,kObs,t,dt in progbar(chrono.forecast_range):
    E[k] = f.model(E[k-1],t-dt,dt)
    E[k] = add_noise(E[k], dt, f.noise, config)

    if kObs is not None:
      stats.assess(k,kObs,'f',E=E[k])

      kLag     = find_1st_ind(chrono.tt >= t-config.tLag)
      kkLag    = range(kLag, k+1)
      ELag     = E[kkLag]

      hE       = h.model(E[k],t)
      y        = yy[kObs]

      ELag     = reshape_to(ELag)
      ELag     = EnKF_analysis(ELag,hE,h.noise,y,upd_a,stats,kObs)
      E[kkLag] = reshape_fr(ELag,f.m)
      post_process(E[k],config)
      stats.assess(k,kObs,'a',E=E[k])

  for k in progbar(range(chrono.K+1),desc='Assessing'):
    stats.assess(k,None,'u',E=E[k])

  return stats


def EnRTS(setup,config,xx,yy):
  """
  EnRTS (Rauch-Tung-Striebel) smoother.

  Ref: Raanes, Patrick Nima. (2016):
  "On the ensemble Rauch‐Tung‐Striebel smoother..."

  Settings for reproducing literature benchmarks may be found in
  mods/Lorenz95/raanes2016.py
  """

  f,h,chrono,X0 = setup.f, setup.h, setup.t, setup.X0
  N, upd_a      = config.N, config.upd_a

  E     = zeros((chrono.K+1,N,f.m))
  Ef    = E.copy()
  E[0]  = X0.sample(N)
  stats = Stats(setup,config,xx,yy)

  # Forward pass
  for k,kObs,t,dt in progbar(chrono.forecast_range):
    E[k]  = f.model(E[k-1],t-dt,dt)
    E[k]  = add_noise(E[k], dt, f.noise, config)
    Ef[k] = E[k]

    if kObs is not None:
      stats.assess(k,kObs,'f',E=E[k])
      hE   = h.model(E[k],t)
      y    = yy[kObs]
      E[k] = EnKF_analysis(E[k],hE,h.noise,y,upd_a,stats,kObs)
      post_process(E[k],config)
      stats.assess(k,kObs,'a',E=E[k])

  # Backward pass
  for k in progbar(range(chrono.K)[::-1]):
    A  = anom(E[k])[0]
    Af = anom(Ef[k+1])[0]

    J = tinv(Af) @ A
    J *= config.cntr
    
    E[k] += ( E[k+1] - Ef[k+1] ) @ J

  for k in progbar(range(chrono.K+1),desc='Assessing'):
    stats.assess(k,E=E[k])
  return stats



def add_noise(E, dt, noise, config):
  """
  Treatment of additive noise for ensembles.
  Settings for reproducing literature benchmarks may be found in
  mods/LA/raanes2015.py
  """
  method = getattr(config,'fnoise_treatm','Stoch')

  if not noise.is_random: return E

  N,m  = E.shape
  A,mu = anom(E)
  Q12  = noise.C.ssqrt
  Q    = noise.C.C

  def sqrt_core():
    T    = np.nan
    Qa12 = np.nan
    A2   = A.copy() # Instead of using nonlocal A, which changes A
    # in outside scope as well. NB: This is a bug in Datum!
    if N<=m:
      Ainv = tinv(A2.T)
      Qa12 = Ainv@Q12
      T    = funm_psd(eye(N) + dt*(N-1)*(Qa12@Qa12.T), sqrt)
      A2   = T@A2
    else: # "Left-multiplying" form
      P = A2.T @ A2 /(N-1)
      L = funm_psd(eye(m) + dt*mrdiv(Q,P), sqrt)
      A2= A2 @ L.T
    E = mu + A2
    return E, T, Qa12

  if method == 'Stoch':
    # In-place addition works (also) for empty [] noise sample.
    E += sqrt(dt)*noise.sample(N)
  elif method == 'none':
    pass
  elif method == 'Mult-1':
    varE   = np.var(E,axis=0,ddof=1).sum()
    ratio  = (varE + dt*diag(Q).sum())/varE
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


def EnKF_analysis(E,hE,hnoise,y,upd_a,stats,kObs):
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
          trHK  = np.sum( (s**2+(N-1))**(-1.0) * s**2 ) # see docs/trHK.jpg
        elif 'sS' in upd_a:
          # Same as 'svd', but with slightly different notation
          # (sometimes used by Sakov) using the normalization sqrt(N-1).
          #z    = dy@ R.m12.T / sqrt(N-1)
          S     = Y @ R.m12.T / sqrt(N-1)
          V,s,_ = svd0(S)
          d     = pad0(s**2,N) + 1
          Pw    = ( V * d**(-1.0) )@V.T / (N-1) # = G/(N-1)
          T     = ( V * d**(-0.5) )@V.T
          trHK  = np.sum(  (s**2 + 1)**(-1.0)*s**2 ) # see docs/trHK.jpg
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
      raise KeyError("No analysis update method found: '" + upd_a + "'.") 

    # Diagnostic: relative influence of observations
    if 'trHK' in locals(): stats.trHK[kObs] = trHK     /hnoise.m
    elif 'HK' in locals(): stats.trHK[kObs] = trace(HK)/hnoise.m

    return E


def serial_inds(upd_a, y, cvR, A):
  if 'mono' in upd_a:
    # Not robust?
    inds = arange(len(y))
  elif 'sorted' in upd_a:
    dC = diag(cvR.C)
    if np.all(dC == dC[0]):
      # Sort y by P
      dC = np.sum(A*A,0)/(N-1)
    inds = np.argsort(dC)
  else: # Default: random ordering
    inds = np.random.permutation(len(y))
  return inds
  

def SL_EAKF(setup,config,xx,yy):
  """
  Serial, covariance-localized EAKF.

  Ref: Karspeck, Alicia R., and Jeffrey L. Anderson. (2007):
  "Experimental implementation of an ensemble adjustment filter..."

  Used without localization, this should be equivalent
  (full ensemble equality) to the EnKF 'Serial'.
  See DAPPER/Misc/batch_vs_serial.py for some details.
  """
  f,h,chrono,X0,N = setup.f, setup.h, setup.t, setup.X0, config.N
  AMet  = getattr(config,'upd_a','default')
  taper = getattr(config,'taper',None)

  n = N-1

  R    = h.noise
  Rm12 = h.noise.C.m12
  #Ri   = h.noise.C.inv

  E     = X0.sample(N)
  stats = Stats(setup,config,xx,yy).assess(0,E=E)

  for k,kObs,t,dt in progbar(chrono.forecast_range):
    E = f.model(E,t-dt,dt)
    E = add_noise(E, dt, f.noise, config)

    if kObs is not None:
      stats.assess(k,kObs,'f',E=E)
      y    = yy[kObs]
      inds = serial_inds(AMet, y, R, anom(E)[0])
          
      locf_at = h.loc_f(config.loc_rad, 'y2x', t, taper)
      for i,j in enumerate(inds):
        hE = h.model(E,t)
        hx = mean(hE,0)
        Y  = (hE - hx).T
        mu = mean(E ,0)
        A  = E-mu

        # Update j-th component of observed ensemble
        Yj    = Rm12[j,:] @ Y
        dyj   = Rm12[j,:] @ (y - hx)
        #
        skk   = Yj@Yj
        su    = 1/( 1/skk + 1/n )
        alpha = (n/(n+skk))**(0.5)
        #
        dy2   = su*dyj/n # (mean is absorbed in dyj)
        Y2    = alpha*Yj

        if skk<1e-9: continue

        # Update state (regression), with localization
        # Localize
        local, coeffs = locf_at(j)
        if len(local) == 0: continue
        Regression    = (A[:,local]*coeffs).T @ Yj/np.sum(Yj**2)
        mu[ local]   += Regression*dy2
        A[:,local]   += np.outer(Y2 - Yj, Regression)

        # Without localization:
        #Regression = A.T @ Yj/np.sum(Yj**2)
        #mu        += Regression*dy2
        #A         += np.outer(Y2 - Yj, Regression)

        E = mu + A

      post_process(E,config)

    stats.assess(k,kObs,E=E)
  return stats



def LETKF(setup,config,xx,yy):
  """
  Same as EnKF (sqrt), but with localization.

  Settings for reproducing literature benchmarks may be found in
  mods/Lorenz95/sak08.py

  Ref: Hunt, Brian R., Eric J. Kostelich, and Istvan Szunyogh. (2007):
  "Efficient data assimilation for spatiotemporal chaos..."
  """

  f,h,chrono,X0,N = setup.f, setup.h, setup.t, setup.X0, config.N
  upd_a = getattr(config,'upd_a','default')
  taper = getattr(config,'taper',None)

  Rm12 = h.noise.C.m12

  E     = X0.sample(N)
  stats = Stats(setup,config,xx,yy).assess(0,E=E)

  for k,kObs,t,dt in progbar(chrono.forecast_range):
    E = f.model(E,t-dt,dt)
    E = add_noise(E, dt, f.noise, config)

    if kObs is not None:
      stats.assess(k,kObs,'f',E=E)
      mu = mean(E,0)
      A  = E - mu

      hE = h.model(E,t)
      hx = mean(hE,0)
      YR = (hE-hx)  @ Rm12.T
      yR = (yy[kObs] - hx) @ Rm12.T

      locf_at = h.loc_f(config.loc_rad, 'x2y', t, taper)
      for i in range(f.m):
        # Localize
        local, coeffs = locf_at(i)
        if len(local) == 0: continue
        iY  = YR[:,local] * sqrt(coeffs)
        idy = yR[local]   * sqrt(coeffs)

        # Do analysis
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
        stats.trHK[kObs] = (sd**(-1.0) * sd**2).sum()/h.noise.m
      #else:
        # nevermind

    stats.assess(k,kObs,E=E)
  return stats


# Notes:
#  Using minimize_scalar:
#  - don't accept dJdx. Pro: only need J  :-)
#  - method='bounded' not necessary and slower than 'brent'.
#  - bracket not necessary either...
#  Using multivariate minimization: fmin_cg, fmin_bfgs, fmin_ncg
#  - these also accept dJdx. But only fmin_bfgs approaches
#    the speed of the scalar minimizers.
#  Using scalar root-finders:
#  - brenth(dJ1, LowB, 1e2,     xtol=1e-6) # Same speed as minimmization
#  - newton(dJ1,1.0, fprime=dJ2, tol=1e-6) # No improvement
#  - newton(dJ1,1.0, fprime=dJ2, tol=1e-6, fprime2=dJ3) # No improvement
#  - myNewton(dJ1,dJ2, 1.0) # Significantly faster. Also slightly better CV?
# => Despite inconvienience of defining analytic derivatives,
#    myNewton seems like the best option.
# NB: In extreme (or just non-linear h) cases,
#     the EnKF-N cost function may have multiple minima.
#     Then: must use more robust optimizer!
def myNewton(fun,deriv,x0,conf=1.0,xtol=1e-4,itermax=10**4):
  "Simple implementation of Newton root-finding"
  x       = x0
  itr     = 0
  dx      = np.inf
  while True and xtol<dx and itr<itermax:
    Jx = fun(x)
    Dx = deriv(x)
    dx = Jx/Dx * conf
    x -= dx
  return x

def EnKF_N(setup,config,xx,yy):
  """
  Finite-size EnKF (EnKF-N) -- dual formulation.

  Dual ≡ Primal if using the Hessian adjustment, and lklhd is Gaussian.
  
  Note: Implementation corresponds to version ql2 of Datum.

  Ref: Bocquet, Marc, Patrick N. Raanes, and Alexis Hannart. (2015):
  "Expanding the validity of the ensemble Kalman filter..."

  Settings for reproducing literature benchmarks may be found in
  mods/Lorenz95/sak08.py
  mods/Lorenz95/sak12.py
  """

  # Unpack
  f,h,chrono,X0  = setup.f, setup.h, setup.t, setup.X0
  N, primal_Hess = config.N, getattr(config,'Hess',False)

  Rm12 = h.noise.C.m12
  Ri   = h.noise.C.inv

  # EnKF-N constants
  g    = 1             # Nullity of Y (obs anom's).
  #g   = max(1,N-h.m)  # TODO: No good
  eN   = (N+1)/N       # Effect of unknown mean
  clog = (N+g)/(N-1)   # Coeff in front of log term
  mode = eN/clog       # Mode of prior for lambda
  LowB = sqrt(mode)    # Lower bound for lambda^1

  E     = X0.sample(N)
  stats = Stats(setup,config,xx,yy).assess(0,E=E)

  for k,kObs,t,dt in progbar(chrono.forecast_range):
    E = f.model(E,t-dt,dt)
    E = add_noise(E, dt, f.noise, config)

    if kObs is not None:
      stats.assess(k,kObs,'f',E=E)
      hE = h.model(E,t)
      y  = yy[kObs]

      mu = mean(E,0)
      A  = E - mu

      hx = mean(hE,0)
      Y  = hE-hx
      dy = y - hx

      V,s,U_T = svd0( Y @ Rm12.T )

      # Make dual cost function (in terms of lambda^1)
      m_Nm = min(N,h.m)
      du   = U_T @ (Rm12 @ dy)
      dgn  = lambda l: pad0( (l*s)**2, m_Nm ) + (N-1)
      PR   = (s**2).sum()/(N-1)
      fctr = sqrt(mode**(1/(1+PR)))
      J    = lambda l:          np.sum(du**2/dgn(l)) \
             + (1/fctr)*eN/l**2 \
             + fctr*clog*log(l**2)
      #l1  = sp.optimize.minimize_scalar(J, bracket=(LowB, 1e2), tol=1e-4).x
      # Derivatives
      dJ1  = lambda l: -2*l   * np.sum(pad0(s**2, m_Nm) * du**2/dgn(l)**2) \
             + -2*(1/fctr)*eN/l**3 \
             +  2*fctr*clog  /l
      dJ2  = lambda l: 8*l**2 * np.sum(pad0(s**4, m_Nm) * du**2/dgn(l)**3) \
             +  6*(1/fctr)*eN/l**4 \
             + -2*fctr*clog  /l**2
      # Find inflation factor
      l1 = myNewton(dJ1,dJ2, 1.0)

      # Turns EnKF-N into ETKF:
      #l1 = 1.0

      # Inflate prior.
      A *= l1
      Y *= l1

      # Compute ETKF (sym sqrt) update
      dgn     = lambda l: pad0( (l*s)**2, N ) + (N-1)
      Pw      = (V * dgn(l1)**(-1.0)) @ V.T
      w       = dy@Ri@Y.T@Pw

      if primal_Hess:
        zeta  = (N-1)/l1**2
        Hess  = Y@Ri@Y.T + zeta*eye(N) - 2*zeta**2/(N+g)*np.outer(w,w)
        T     = funm_psd(Hess, lambda x: x**-.5) * sqrt(N-1) # sqrtm Woodbury?
      else:
        T     = (V * dgn(l1)**(-0.5)) @ V.T * sqrt(N-1)
        
      E = mu + w@A + T@A
      post_process(E,config)

      stats.infl[kObs] = l1
      stats.trHK[kObs] = (((l1*s)**2 + (N-1))**(-1.0)*s**2).sum()/h.noise.m

    stats.assess(k,kObs,E=E)
  return stats



def iEnKF(setup,config,xx,yy):
  """
  Iterative EnKS.

  Loosely adapted from Bocquet ienks code and 
  Ref:.Bocquet, Marc, and Pavel Sakov. (2014):
  "An iterative ensemble Kalman smoother."

  Settings for reproducing literature benchmarks may be found in
  mods/Lorenz95/sak08.py
  mods/Lorenz95/sak12.py
  """
  f,h,chrono,X0,R = setup.f, setup.h, setup.t, setup.X0, setup.h.noise.C
  N, upd_a        = config.N, config.upd_a

  E           = X0.sample(N)
  stats       = Stats(setup,config,xx,yy)
  stats.iters = np.full(chrono.KObs+1,nan)
  stats.assess(0,E=E)

  for kObs in progbar(range(chrono.KObs+1)):
    xb0 = mean(E,0)
    A0  = E - xb0
    # Init
    w      = zeros(N)
    Tinv   = eye(N)
    T      = eye(N)
    for iteration in range(config.iMax):
      E = xb0 + w @ A0 + T @ A0
      for k,t,dt in chrono.DAW_range(kObs):
        E = f.model(E,t-dt,dt)
        E = add_noise(E, dt, f.noise, config)
      if iteration==0:
        stats.assess(k,kObs,'f',E=E)
  
      hE = h.model(E,t)
      hx = mean(hE,0)
      Y  = hE-hx
      Y  = Tinv @ Y
      y  = yy[kObs]
      dy = y - hx

      dw,Pw,T,Tinv = iEnKF_analysis(w,dy,Y,h.noise,upd_a)
      w  -= dw
      if np.linalg.norm(dw) < N*1e-4:
        break

    stats.trHK [kObs] = trace(R.inv @ Y.T @ Pw @ Y)/h.noise.m
    stats.iters[kObs] = iteration+1

    E = xb0 + w @ A0 + T @ A0
    post_process(E,config)

    for k,t,dt in chrono.DAW_range(kObs):
      E = f.model(E,t-dt,dt)
      E = add_noise(E, dt, f.noise, config)
      stats.assess(k,None,'u',E=E)
    stats.assess(k,kObs,'a',E=E)

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
    if 'explicit' in upd_a:
      Pw   = funm_psd(hess, np.reciprocal)
      T    = funm_psd(hess, lambda x: x**(-0.5)) * sqrt(N-1)
      Tinv = funm_psd(hess, sqrt) / sqrt(N-1)
    elif 'svd' in upd_a:
      # Implementation using svd of Y
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



import scipy.optimize as opt
def BnKF(setup,config,xx,yy):

  # Test settings:
  #from mods.Lorenz95.sak08 import setup
  #config = DAC(EnKF_N,N=28)
  #config = DAC(EnKF,'PertObs',N=39,infl=1.06)
  #config = DAC(BnKF,N=39,infl=1.03)

  # Unpack
  f,h,chrono,X0  = setup.f, setup.h, setup.t, setup.X0

  Rm12 = h.noise.C.m12
  Ri   = h.noise.C.inv

  # constants
  N  = config.N
  nu = N-1

  invm = lambda x: funm_psd(x, np.reciprocal)
  IN  = eye(N)
  Pi1 = np.outer(ones(N),ones(N))/N
  PiC = IN - Pi1

  # Init
  bb    = N*ones(N)
  E     = X0.sample(N)
  stats = Stats(setup,config,xx,yy).assess(0,E=E)

  for k,kObs,t,dt in progbar(chrono.forecast_range):
    E = f.model(E,t-dt,dt)
    E = add_noise(E, dt, f.noise, config)

    if kObs is not None:
      stats.assess(k,kObs,'f',E=E)
      hE = h.model(E,t)
      y  = yy[kObs]

      mu = mean(E,0)
      A  = E - mu

      hx = mean(hE,0)
      Y  = hE-hx
      dy = y - hx

      #  V,s,U_T = svd0( Y @ Rm12.T )

      #  # Compute ETKF (sym sqrt) update
      #  l1      = 1.0
      #  dgn     = lambda l: pad0( (l*s)**2, N ) + (N-1)
      #  Pw      = (V * dgn(l1)**(-1.0)) @ V.T
      #  w       = dy@Ri@Y.T@Pw
      #  T       = (V * dgn(l1)**(-0.5)) @ V.T * sqrt(N-1)

      #  E = mu + w@A + T@A

      # Prepare
      V,s,_  = svd0( Y @ Rm12.T ) 
      target = invm( Y@Ri@Y.T/nu + eye(N) ) # = nu*Pwn

      dC = np.zeros((N,N))
      def resulting_Pw(rr):
        for n in arange(N):
          dn      = y-hE[n]
          an      = nu*rr[n]/bb[n]
          #xn    += A@Y.T @ invm( Y@Y.T + an*R ) @ (y-xn+noise)
          #Pwn    = invm( Y.T @ Ri @ Y/an + eye(N))/an
          dgn     = pad0(s**2,N) + an
          Pwn     = ( V * dgn**(-1.0) ) @ V.T
          dC[:,n] = IN[:,n] + Pwn@Y@Ri@dn
        Ca = dC @ PiC @ dC.T
        return Ca

      def inpT(logr):
        assert len(logr)==(N-1)
        rr = np.hstack([exp(logr), N])
        rr*= np.sum(1/rr)
        return rr

      def r2(x):
        rr = inpT(x[:-1])
        Ca = resulting_Pw(rr)
        diff = diag(  target - (Ca + x[-1]*Pi1)  )
        return diff

      x0  = np.hstack([log(N*ones(N-1)), 1])
      sol = opt.root(r2, x0, method='lm', options={'maxiter':1000})
      rr_ = inpT(sol.x[:-1])

      #print(inpT(sol.x[:-1]))
      #print(r2(sol.x))

      # Get PertObs-EnKF
      #D   = center(h.noise.sample(N))
      #rr_ = N*ones(N)
        
      for n in arange(N):
        dn      = y-hE[n] # + D[n]
        an      = nu*rr_[n]/bb[n]
        dg0     = pad0(s**2,N) + nu
        dgn     = pad0(s**2,N) + an
        Pwn     = ( V * dgn**(-1.0) ) @ V.T
        E[n]   += A.T@Pwn@Y@Ri@dn
        bb[n]   = mean(dg0/dgn*rr_[n])
      bb *= np.sum(1/bb)

      post_process(E,config)
    stats.assess(k,kObs,E=E,w=1/bb) # TODO
  return stats





def PartFilt(setup,config,xx,yy):
  """
  Particle filter ≡ Sequential importance (re)sampling SIS (SIR).
  This is the bootstrap version: the proposal density is just
  q(x_0:t|y_1:t) = p(x_0:t) = p(x_t|x_{t-1}) p(x_0:{t-1}).

  Ref:
  [1]: Christopher K. Wikle, L. Mark Berliner, 2006:
    "A Bayesian tutorial for data assimilation"
  [2]: Van Leeuwen, 2009: "Particle Filtering in Geophysical Systems"

  Tuning settings:
   - NER: Trigger resampling whenever N_eff <= N*NER.
       If resampling with some variant of 'Multinomial', no systematic bias is introduced.
   - qroot: "Inflate" (anneal) the proposal noise kernels by this root to increase diversity.
       The weights are updated to maintain un-biased-ness.
       Ref: [3], section VI-M.2

  [3]: Zhe Chen, 2003:
    "Bayesian Filtering: From Kalman Filters to Particle Filters, and Beyond"
  """

  f,h,chrono,X0 = setup.f, setup.h, setup.t, setup.X0
  N, upd_a      = config.N, getattr(config,'upd_a',None)
  NER           = config.NER
  wroot         = getattr(config,'wroot',1.0)
  qroot         = getattr(config,'qroot',1.0)
  reg           = getattr(config,'reg',0)
  rroot         = getattr(config,'rroot',1.0)
  nuj           = getattr(config,'nuj',False)

  Rm12 = h.noise.C.m12
  Q12  = f.noise.C.ssqrt

  E = X0.sample(N)
  w = 1/N*ones(N)

  stats        = Stats(setup,config,xx,yy)
  stats.N_eff  = np.full(chrono.KObs+1,nan)
  stats.resmpl = zeros(chrono.KObs+1,dtype=bool)
  stats.innovs = np.full((chrono.KObs+1,N,h.m),nan)
  stats.assess(0,E=E,w=1/N)

  for k,kObs,t,dt in progbar(chrono.forecast_range):
    E  = f.model(E,t-dt,dt)
    D  = randn((N,f.m))
    E += sqrt(dt*qroot)*(D@Q12.T)

    if qroot != 1.0:
      # Evaluate p/q (at each noise realization in D) when q:=p**(1/qroot).
      w *= exp(-0.5*np.sum(D**2, axis=1) * (1 - 1/qroot))
      w /= w.sum()

    if kObs is not None:
      stats.assess(k,kObs,'f',E=E,w=w)

      hE = h.model(E,t)
      y  = yy[kObs]
      innovs = hE - y
      innovs = innovs @ Rm12.T
      logL   = -0.5 * np.sum(innovs**2, axis=1)
      w      = reweight(w,logL=logL)

      stats.innovs[kObs] = innovs

      if trigger_resampling(w,NER,stats,kObs):
        E,w = resample(E, w, kind=upd_a, reg=reg, wroot=wroot, rroot=rroot, no_uniq_jitter=nuj)

      post_process(E,config)
    stats.assess(k,kObs,E=E,w=w)
  return stats


def OptPF(setup,config,xx,yy):
  """
  "Optimal proposal" particle filter.
  OR
  Implicit particle filter.

  Note: Regularization is here added BEFORE Bayes' rule.

  Ref: Bocquet et al. (2010):
    "Beyond Gaussian statistical modeling in geophysical data assimilation"
  """

  f,h,chrono,X0 = setup.f, setup.h, setup.t, setup.X0
  N, upd_a      = config.N, getattr(config,'upd_a',None)
  NER           = config.NER
  Qs            = getattr(config,'Qs',0)
  reg           = getattr(config,'reg',0)
  nuj           = getattr(config,'nuj',False)

  R    = h.noise.C.C
  Rm12 = h.noise.C.m12
  Q12  = f.noise.C.ssqrt

  E = X0.sample(N)
  w = 1/N*ones(N)

  stats        = Stats(setup,config,xx,yy)
  stats.N_eff  = np.full(chrono.KObs+1,nan)
  stats.resmpl = zeros(chrono.KObs+1,dtype=bool)
  stats.assess(0,E=E,w=1/N)

  for k,kObs,t,dt in progbar(chrono.forecast_range):
    E  = f.model(E,t-dt,dt)
    E += sqrt(dt)*(randn((N,f.m))@Q12.T)

    if kObs is not None:
      stats.assess(k,kObs,'f',E=E,w=w)

      y      = yy[kObs]
      innovs = y - h.model(E,t) # Note: before adding noise

      mu   = w@E
      A    = E - mu
      ub   = unbias_var(w, avoid_pathological=True)
      bw   = N**(-1/(f.m+4))
      nrm  = tp(sqrt(Qs*bw*ub*w)) # ≈ 1/sqrt(N-1)
      E   += sample_quickly_with(nrm*A)[0]

      hE  = h.model(E,t)
      hx  = w@hE
      Y   = hE-hx

      D    = center(h.noise.sample(N))
      C    = (nrm*Y).T @ (nrm*Y) + R
      dE   = (nrm*A).T @ (nrm*Y) @ mldiv(C,(y-hE+D).T)
      E    = E + dE.T

      chi2   = innovs*mldiv(C,innovs.T).T
      logL   = -0.5 * np.sum(chi2, axis=1)
      w      = reweight(w,logL=logL)

      if trigger_resampling(w,NER,stats,kObs):
        E,w = resample(E, w, kind=upd_a, reg=reg, no_uniq_jitter=nuj)

      post_process(E,config)
    stats.assess(k,kObs,E=E,w=w)
  return stats



def trigger_resampling(w,NER,stats,kObs):
  "Resample if N_effective <= threshold."
  N_eff              = 1/(w@w)
  do_resample        = N_eff <= len(w)*NER
  stats.N_eff[kObs]  = N_eff
  stats.resmpl[kObs] = do_resample
  return do_resample

def reweight(w,lklhd=None,logL=None):
  """
  Do Bayes' rule for the empirical distribution of an importance sample.
  Do computations in log-space, for at least 2 reasons:
  - Normalization: will fail if sum==0 (if all innov's are large).
  - Num. precision: lklhd*w should have better prec in log space.
  Output is non-log, for the purpose of assessment and resampling.
  """
  if lklhd is not None:
    logL = log(lklhd)
  logL  -= logL.max()    # Avoid numerical error
  logw   = log(w) + logL # Bayes' rule
  w      = exp(logw)
  w     /= w.sum()
  return w


def resample(E,w,N=None,kind=None,
    fix_mu=False,fix_var=False,reg=0.0,wroot=1.0,rroot=1.0,no_uniq_jitter=False):
  """
  Resampling function for the particle filter.

  Example: see docs/test_resample.py.

  - N can be different from E.shape[0]
  (e.g. in case some particles have been elimintated).

  - kind: 'Residual' and 'Systematic' more systematic (less stochastic)
    variations on 'Multinomial' sampling. 'Systematic' is a little faster.
    'Gaussian' is biased (approximate) unless the actual distribution is Gaussian.

  After resampling some of the particles will be identical.
  Therefore, if noise.is_deterministic: some noise must be added.
  This is adjusted by the regularization 'reg' factor
  (so-named because Dirac-deltas are approximated  Gaussian kernels),
  which controls the strength of the jitter.
  This causes a bias. But, as N-->∞, the reg. bandwidth-->0, i.e. bias-->0.
  Ref: [3], section 12.2.2.

  - wroot: Adjust weights before resampling by this root to mitigate thinning.
       The outcomes of the resampling are then weighted to maintain un-biased-ness.
       Ref: [4], section 3.1

  Note: (a) resampling methods are beneficial because they discard
  low-weight ("doomed") particles and reduce the variance of the weights.
  However, (b) even unbiased/rigorous resampling methods introduce noise;
  (increases the var of any empirical estimator, see [1], section 3.4).
  How to unify the seemingly contrary statements of (a) and (b) ?
  By recognizing that we're in the *sequential/dynamical* setting,
  and that *future* variance may be expected to be lower by focusing
  on the high-weight particles which we anticipate will 
  have more informative (and less variable) future likelihoods.

  Note: anomalies (and thus cov) are weighted,
  and also computed based on a weighted mean.

  [1]: Doucet, Johansen, 2009, v1.1:
    "A Tutorial on Particle Filtering and Smoothing: Fifteen years later." 
  [2]: Van Leeuwen, 2009: "Particle Filtering in Geophysical Systems"
  [3]: Doucet, de Freitas, Gordon, 2001:
    "Sequential Monte Carlo Methods in Practice"
  [4]: Liu, Chen Longvinenko, 2001:
    "A theoretical framework for sequential importance sampling with resampling"
  """
  assert(abs(w.sum()-1) < 1e-5)

  # TODO: fix_mu, fix_var. Leave note.


  N_o,m = E.shape

  # Input parsing
  if N is None:
    N = N_o
  if kind is None:
    kind = 'Systematic'

  # Stats of original sample that may get used
  if kind is 'Gaussian' or reg!=0 or fix_mu or fix_var:
    mu_o  = w@E
    A_o   = E - mu_o
    ss_o  = sqrt(w @ A_o**2)
    ub    = unbias_var(w, avoid_pathological=True)
    Colr  = tp(sqrt(ub*w)) * A_o

  if kind is 'Gaussian':
    assert wroot==1.0
    w = 1/N*ones(N)
    E = mu_o + sample_quickly_with(Colr)[0]
  else:
    # Compute factors s such that s*w := w**(1/wroot). 
    if wroot!=1.0:
      s   = ( w**(1/wroot - 1) ).clip(max=1e100)
      s  /= (s*w).sum()
      sw  = s*w
    else:
      s   = ones(N_o)
      sw  = w

    if kind is 'Multinomial':
      # van Leeuwen [2] also calls this "probabilistic" resampling
      idx = np.random.choice(N_o,N,replace=True,p=sw)
    elif kind is 'Residual':
      # Doucet [1] also calls this "stratified" resampling.
      w_N   = sw*N            # upscale
      w_I   = w_N.astype(int) # integer part
      w_D   = w_N-w_I         # decimal part
      # Create duplicate indices for integer parts
      idx_I = [i*ones(wi,dtype=int) for i,wi in enumerate(w_I)]
      idx_I = np.concatenate(idx_I)
      # Multinomial sampling of decimal parts
      N_I   = w_I.sum() # == len(idx_I)
      N_D   = N - N_I
      idx_D = np.random.choice(N_o,N_D,replace=True,p=w_D/w_D.sum())
      # Concatenate
      idx   = np.hstack((idx_I,idx_D))
    elif kind is 'Systematic':
      # van Leeuwen [2] also calls this "stochastic universal" resampling
      U     = rand(1) / N
      CDF_a = U + arange(N)/N
      CDF_o = np.cumsum(sw)
      #idx  = CDF_a <= CDF_o[:,None]
      #idx  = np.argmax(idx,axis=0) # Finds 1st. SO/a/16244044/
      idx   = np.searchsorted(CDF_o,CDF_a)
    else:
      raise KeyError

    E  = E[idx]
    w  = 1/s[idx]
    w /= w.sum()

    # Add noise (jittering)
    if reg!=0:
      bw    = N**(-1/(m+4)) # Scott's rule-of-thumb
      scale = reg*bw*sqrt(rroot)
      # Jitter
      if no_uniq_jitter:
        assert kind is 'Systematic'
        dups  = idx == np.roll(idx,1)
        dups |= idx == np.roll(idx,-1)
        sample, chi2 = sample_quickly_with(Colr,N=sum(dups))
        E[dups] += scale*sample
      else:
        sample, chi2 = sample_quickly_with(Colr,N=E.shape[0])
        E += scale*sample
      # Compensate for using root
      if rroot != 1.0:
        w *= exp(-0.5*chi2*(1 - 1/rroot))
        w /= w.sum()

  # While multinomial sampling is unbiased, it does incur sampling error.
  # fix_mu (and fix_var) compensates for this "in the mean" (and variance).
  # Warning: while an interesting concept, this is likely not beneficial.
  A_a,mu_a = anom(E)
  if fix_mu:
    mu_a = mu_o
  if fix_var:
    var_o = (ss_o**2).sum()/m
    var_a = (A_a**2) .sum()/(N*m)
    A_a  *= sqrt(var_o/var_a)
  if fix_mu or fix_var:
    E = mu_a + A_a
    
  return E, w


def sample_quickly_with(Colour,N=None):
  """Gaussian, coloured sampling in the quickest fashion,
  which depends on the size of the colouring matrix."""
  (N_,m) = Colour.shape
  if N is None: N = N_
  if N_ > 2*m:
    cholU  = chol_trunc(Colour.T@Colour)
    D      = randn((N,cholU.shape[0]))
    chi2   = np.sum(D**2, axis=1)
    sample = D@cholU
  else:
    chi2_compensate_for_rank = min(m/N_,1.0)
    D      = randn((N,N_))
    chi2   = np.sum(D**2, axis=1) * chi2_compensate_for_rank
    sample = D@Colour
  return sample, chi2  


def PFD(setup,config,xx,yy):
  """
  Idea: sample a bunch from each kernel.
  => The ones more likely to get picked by resampling are closer to the likelihood.

  Additional idea: employ w-adjustment to obtain N unique particles, without jittering.
  """

  f,h,chrono,X0 = setup.f, setup.h, setup.t, setup.X0
  N, upd_a      = config.N, getattr(config,'upd_a',None)
  NER           = config.NER
  m             = f.m
  Qs            = getattr(config,'Qs',0)
  Qsroot        = getattr(config,'Qsroot',1.0)
  reg           = getattr(config,'reg',0)

  xN            = getattr(config,'xN',False)
  DD            = None

  Rm12 = h.noise.C.m12
  Q12  = f.noise.C.ssqrt

  E = X0.sample(N)
  w = 1/N*ones(N)

  stats        = Stats(setup,config,xx,yy)
  stats.N_eff  = np.full(chrono.KObs+1,nan)
  stats.resmpl = zeros(chrono.KObs+1,dtype=bool)
  stats.assess(0,E=E,w=1/N)

  for k,kObs,t,dt in progbar(chrono.forecast_range):
    E  = f.model(E,t-dt,dt)
    E += sqrt(dt)*(randn((N,m))@Q12.T)

    if kObs is not None:
      stats.assess(k,kObs,'f',E=E,w=w)
      y  = yy[kObs]

      hE     = h.model(E,t)
      innovs = hE - y
      innovs = innovs @ Rm12.T
      logL   = -0.5 * np.sum(innovs**2, axis=1)
      w_     = w.copy()
      w      = reweight(w,logL=logL)

      if trigger_resampling(w,NER,stats,kObs):
        w    = w_

        mu   = w@E
        A    = E - mu
        ub   = unbias_var(w, avoid_pathological=True)
        bw   = N**(-1/(m+4))
        nrm  = tp(sqrt(Qs*bw*ub*w)) # ≈ 1/sqrt(N-1)

        ED   = E.repeat(xN,0)
        wD   = w.repeat(xN)
        wD  /= wD.sum()

        Color = nrm*A
        cholU = chol_trunc(Color.T@Color)
        rnk   = cholU.shape[0]
        if DD is None:
          DD  = randn((N*xN,m))
        ED   += DD[:,:rnk]@cholU
        
        hE     = h.model(ED,t)
        innovs = hE - y
        innovs = innovs @ Rm12.T
        logL   = -0.5 * np.sum(innovs**2, axis=1)
        wD     = reweight(wD,logL=logL)

        E,w = resample(ED, wD, N=N, kind=upd_a, reg=0)

        # Add noise (jittering)
        if reg!=0:
          bw    = N**(-1/(m+4))
          scale = reg*bw
          D, _  = sample_quickly_with(anom(E)[0]/sqrt(N-1))
          E    += scale*D

      post_process(E,config)
    stats.assess(k,kObs,E=E,w=w)
  return stats



def EnCheat(setup,config,xx,yy):
  """
  A baseline/reference method.
  Ensemble method that cheats: it knows the truth.
  Nevertheless, its error will not necessarily be 0,
  because the truth may be outside of the ensemble subspace.
  This method is just to provide a baseline for comparison with other methods.
  It may very well beat the particle filter with N=infinty.
  NB: The forecasts (and their rmse) are given by the standard EnKF.
  """

  f,h,chrono,X0 = setup.f, setup.h, setup.t, setup.X0
  N, upd_a      = config.N, config.upd_a

  E     = X0.sample(N)
  stats = Stats(setup,config,xx,yy).assess(0,E=E)

  for k,kObs,t,dt in progbar(chrono.forecast_range):
    E = f.model(E,t-dt,dt)
    E = add_noise(E, dt, f.noise, config)

    if kObs is not None:
      # Standard EnKF analysis
      hE = h.model(E,t)
      y  = yy[kObs]
      E  = EnKF_analysis(E,hE,h.noise,y,upd_a,stats,kObs)
      post_process(E,config)

      # Cheating (only used for stats)
      w,res,_,_ = sla.lstsq(E.T, xx[k])
      if not res.size:
        res = 0
      res = diag((res/setup.f.m) * ones(setup.f.m))
      opt = w @ E
      # NB: Center on the optimal solution?
      #E += opt - mean(E,0)

    stats.assess(k,kObs,mu=opt,Cov=res)
  return stats


def Climatology(setup,config,xx,yy):
  """
  A baseline/reference method.
  Note that the "climatology" is computed from truth,
  which might be (unfairly) advantageous if this simulation is too short.
  """
  f,h,chrono,X0 = setup.f, setup.h, setup.t, setup.X0

  mu0   = mean(xx,0)
  A0    = xx - mu0
  P0    = spCovMat(A=A0)

  stats = Stats(setup,config,xx,yy).assess(0,mu=mu0,Cov=P0.C)
  stats.trHK[:] = 0

  for k,kObs,_,_ in progbar(chrono.forecast_range):
    stats.assess(k,kObs,'fau',mu=mu0,Cov=P0.C)
  return stats

def D3Var(setup,config,xx,yy):
  """
  3D-Var -- a baseline/reference method.
  Uses the Kalman filter equations,
  but with a prior from the Climatology.
  """
  f,h,chrono,X0 = setup.f, setup.h, setup.t, setup.X0
  infl  = getattr(config,'infl',1.0)
  dkObs = chrono.dkObs
  R     = h.noise.C.C

  mu0   = mean(xx,0)
  A0    = xx - mu0
  P0    = (A0.T @ A0) * (infl/(xx.shape[0] - 1))

  # The uncertainty estimate P is only computed
  # for the sake of the stats, and not for actual DA.
  # Even though it should be beneficial to use P instead of P0
  # in the mu update, that is beyond our scope.
  # I.e. the actual DA only relies on P0.
  P     = P0    # estimate
  r     = dkObs # intra-DAW counter
  # Experimental:
  L       = estimate_corr_length(A0.ravel(order='F'))
  max_amp = 2*P0 # var(mu-truth) = 2 P0
  def saw_tooth(Pa,rho):
    """A sigmoid fitted to decorrelation length (L), Pa, and P0."""
    # See doc/saw_tooth.jpg
    sigmoid = lambda t: 1/(1+exp(-t))
    inv_sig = lambda u: log(u/(1-u))
    shift   = inv_sig(trace(Pa)/trace(max_amp))
    dC      = 0.1 # Correlation increment for comparing corr funcs.
    L_ratio = 2*inv_sig(dC/2) / log(dC)
    fudge   = 1/5.5
    scale   = fudge*dkObs/L*L_ratio
    return max_amp*sigmoid(shift + scale*rho)

  # Try to detect whether KG may be pre-computed by
  # relying on NaN's to trigger errors (if H is time-dep).
  try:
    H  = h.jacob(np.nan, np.nan)
    KG = mrdiv(P0@H.T, H@P0@H.T+R)
    Pa = (eye(f.m) - KG@H) @ P0
    stats.trHK[:] = trace(KG@H)/f.m
    pre_computed_KG = True
  except Exception:
    pre_computed_KG = False

  # Init
  mu    = X0.mu
  stats = Stats(setup,config,xx,yy).assess(0,mu=mu,Cov=P0)
  
  for k,kObs,t,dt in progbar(chrono.forecast_range):
    mu = f.model(mu,t-dt,dt)

    if kObs is not None:
      stats.assess(k,kObs,'f',mu=mu0,Cov=P0)

      r = 0
      y = yy[kObs]
      if not pre_computed_KG:
        H  = h.jacob(mu0,t)
        KG = mrdiv(P0@H.T, H@P0@H.T + R)
        KH = KG@H
        Pa = (eye(f.m) - KH) @ P0
        stats.trHK[kObs] = trace(KH)/f.m
      mu = mu0 + KG@(y - h.model(mu0,t))

    if r<dkObs:
      # Interpolate P between Pa and P0.
      #P = Pa + (max_amp-Pa)*r/dkObs
      P = saw_tooth(Pa,r/dkObs)
    r += 1

    stats.assess(k,kObs,mu=mu,Cov=P)
  return stats


def ExtKF(setup,config,xx,yy):
  """
  The extended Kalman filter.
  A baseline/reference method.

  If everything is linear-Gaussian, this provides the exact solution
  to the Bayesian filtering equations.

  Inflation ('infl') may be specified.
  It defaults to 1.0, which is ideal in the lin-Gauss case.
  It is applied at each dt, with infl_per_dt := inlf**(dt), so that 
  infl_per_unit_time == infl.
  Specifying it this way (per unit time) means less tuning.
  """
  f,h,chrono,X0 = setup.f, setup.h, setup.t, setup.X0

  infl = getattr(config,'infl',1.0)
  
  R = h.noise.C.C
  Q = f.noise.C.C

  mu = X0.mu
  P  = X0.C.C

  stats = Stats(setup,config,xx,yy).assess(0,mu=mu,Cov=P)

  for k,kObs,t,dt in progbar(chrono.forecast_range):
    
    F = f.jacob(mu,t-dt,dt) 
    # "EKF for the mean". Rarely worth the effort. Matlab code:
    # for k = 1:m
    #   HessianF_k = hessianest(@(x) submat(F(t,dt,x), k), X(:,iT-1))
    #   HessCov(k) = sum(sum( HessianF_k .* P(:,:,iT-1) ))
    # X(:,iT) = X(:,iT) + 1/2*HessCov 

    mu = f.model(mu,t-dt,dt)
    P  = infl**(dt)*(F@P@F.T) + dt*Q

    if kObs is not None:
      stats.assess(k,kObs,'f',mu=mu,Cov=P)
      H  = h.jacob(mu,t)
      KG = mrdiv(P @ H.T, H@P@H.T + R)
      y  = yy[kObs]
      mu = mu + KG@(y - h.model(mu,t))
      KH = KG@H
      P  = (eye(f.m) - KH) @ P

      stats.trHK[kObs] = trace(KH)/f.m

    stats.assess(k,kObs,mu=mu,Cov=P)
  return stats




def post_process(E,config):
  """
  Inflate, Rotate.

  To avoid recomputing/recombining anomalies, this should be inside EnKF_analysis().
  But for readability it is nicer to keep it as a separate function,
  also since it avoids inflating/rotationg smoothed states (for the EnKS).
  """
  infl = getattr(config,'infl',1.0)
  rot  = getattr(config,'rot' ,False)

  if infl!=1.0 or rot:
    A, mu = anom(E)
    N,m   = E.shape
    T     = eye(N)

    if infl!=1.0:
      T = infl * T

    if rot:
      T = genOG_1(N) @ T

    E[:] = mu + T@A




