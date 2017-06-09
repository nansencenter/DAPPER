from common import *

@DA_Config
def EnKF(upd_a,N,infl=1.0,rot=False,**kwargs):
  """
  The EnKF.

  Ref: Evensen, Geir. (2009):
  "The ensemble Kalman filter for combined state and parameter estimation."

  Settings for reproducing literature benchmarks may be found in
  mods/Lorenz95/sak08.py
  """
  def assimilator(stats,twin,xx,yy):
    f,h,chrono,X0 = twin.f, twin.h, twin.t, twin.X0

    # Init
    E = X0.sample(N)
    stats.assess(0,E=E)

    # Loop
    for k,kObs,t,dt in progbar(chrono.forecast_range):
      E = f(E,t-dt,dt)
      E = add_noise(E, dt, f.noise, kwargs)

      # Analysis update
      if kObs is not None:
        stats.assess(k,kObs,'f',E=E)
        E = EnKF_analysis(E,h(E,t),h.noise,yy[kObs],upd_a,stats,kObs)
        E = post_process(E,infl,rot)

      stats.assess(k,kObs,E=E)
  return assimilator



@DA_Config
def EnKS(upd_a,N,tLag,infl=1.0,rot=False,**kwargs):
  """
  EnKS (ensemble Kalman smoother)

  Ref: Evensen, Geir. (2009):
  "The ensemble Kalman filter for combined state and parameter estimation."

  The only difference to the EnKF is the management of the lag and the reshapings.
  Settings for reproducing literature benchmarks may be found in
  mods/Lorenz95/raanes2016.py
  """
  def assimilator(stats,twin,xx,yy):
    f,h,chrono,X0 = twin.f, twin.h, twin.t, twin.X0

    def reshape_to(E):
      K,N,m = E.shape
      return E.transpose([1,0,2]).reshape((N,K*m))
    def reshape_fr(E,m):
      N,Km = E.shape
      K    = Km//m
      return E.reshape((N,K,m)).transpose([1,0,2])

    E    = zeros((chrono.K+1,N,f.m))
    E[0] = X0.sample(N)

    for k,kObs,t,dt in progbar(chrono.forecast_range):
      E[k] = f(E[k-1],t-dt,dt)
      E[k] = add_noise(E[k], dt, f.noise, kwargs)

      if kObs is not None:
        stats.assess(k,kObs,'f',E=E[k])

        kLag     = find_1st_ind(chrono.tt >= t-tLag)
        kkLag    = range(kLag, k+1)
        ELag     = E[kkLag]

        hE       = h(E[k],t)
        y        = yy[kObs]

        ELag     = reshape_to(ELag)
        ELag     = EnKF_analysis(ELag,hE,h.noise,y,upd_a,stats,kObs)
        E[kkLag] = reshape_fr(ELag,f.m)
        E[k]     = post_process(E[k],infl,rot)
        stats.assess(k,kObs,'a',E=E[k])

    for k in progbar(range(chrono.K+1),desc='Assessing'):
      stats.assess(k,None,'u',E=E[k])
  return assimilator


@DA_Config
def EnRTS(upd_a,N,cntr,infl=1.0,rot=False,**kwargs):
  """
  EnRTS (Rauch-Tung-Striebel) smoother.

  Ref: Raanes, Patrick Nima. (2016):
  "On the ensemble Rauch‐Tung‐Striebel smoother..."

  Settings for reproducing literature benchmarks may be found in
  mods/Lorenz95/raanes2016.py
  """
  def assimilator(stats,twin,xx,yy):
    f,h,chrono,X0 = twin.f, twin.h, twin.t, twin.X0

    E    = zeros((chrono.K+1,N,f.m))
    Ef   = E.copy()
    E[0] = X0.sample(N)

    # Forward pass
    for k,kObs,t,dt in progbar(chrono.forecast_range):
      E[k]  = f(E[k-1],t-dt,dt)
      E[k]  = add_noise(E[k], dt, f.noise, kwargs)
      Ef[k] = E[k]

      if kObs is not None:
        stats.assess(k,kObs,'f',E=E[k])
        hE   = h(E[k],t)
        y    = yy[kObs]
        E[k] = EnKF_analysis(E[k],hE,h.noise,y,upd_a,stats,kObs)
        E[k] = post_process(E[k],infl,rot)
        stats.assess(k,kObs,'a',E=E[k])

    # Backward pass
    for k in progbar(range(chrono.K)[::-1]):
      A  = anom(E[k])[0]
      Af = anom(Ef[k+1])[0]

      J = tinv(Af) @ A
      J *= cntr
      
      E[k] += ( E[k+1] - Ef[k+1] ) @ J

    for k in progbar(range(chrono.K+1),desc='Assessing'):
      stats.assess(k,E=E[k])
  return assimilator




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
        C  = Y.T @ Y + R.full*(N-1)
        D  = center(hnoise.sample(N))
        YC = mrdiv(Y, C)
        KG = A.T @ YC
        HK = Y.T @ YC
        dE = (KG @ ( y + D - hE ).T).T
        E  = E + dE
    elif 'Sqrt' in upd_a:
        # Uses a symmetric square root (ETKF)
        # to deterministically transform the ensemble.
        #
        # The various versions below differ only numerically.
        # EVD is default, but for large N use SVD version.
        if upd_a == 'Sqrt' and N>m: upd_a = 'Sqrt svd'
        #
        if 'explicit' in upd_a:
          # Not recommended.
          # Implementation using inv (in ens space)
          Pw = inv(Y @ R.inv @ Y.T + (N-1)*eye(N))
          T  = sqrtm(Pw) * sqrt(N-1)
          HK = R.inv @ Y.T @ Pw @ Y
          #KG = R.inv @ Y.T @ Pw @ A
        elif 'svd' in upd_a:
          # Implementation using svd of Y R^{-1/2}.
          V,s,_ = svd0(Y @ R.sym_sqrt_inv.T)
          d     = pad0(s**2,N) + (N-1)
          Pw    = ( V * d**(-1.0) ) @ V.T
          T     = ( V * d**(-0.5) ) @ V.T * sqrt(N-1) 
          trHK  = np.sum( (s**2+(N-1))**(-1.0) * s**2 ) # see docs/trHK.jpg
        elif 'sS' in upd_a:
          # Same as 'svd', but with slightly different notation
          # (sometimes used by Sakov) using the normalization sqrt(N-1).
          S     = Y @ R.sym_sqrt_inv.T / sqrt(N-1)
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
        # Observations assimilator one-at-a-time.
        # Even though it's derived as "serial ETKF",
        # it's not equivalent to 'Sqrt' for the actual ensemble,
        # although it does yield the same mean/cov.
        # See DAPPER/Misc/batch_vs_serial.py for more details.
        inds = serial_inds(upd_a, y, R, A)
        z = dy@ R.sym_sqrt_inv.T / sqrt(N-1)
        S = Y @ R.sym_sqrt_inv.T / sqrt(N-1)
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
        trHK = trace(R.sym_sqrt_inv.T@GS@Y)/sqrt(N-1) # Correct?
    elif 'DEnKF' is upd_a:
        # Uses "Deterministic EnKF" (sakov'08)
        C  = Y.T @ Y + R.full*(N-1)
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



def post_process(E,infl,rot):
  """
  Inflate, Rotate.

  To avoid recomputing/recombining anomalies, this should be inside EnKF_analysis().
  But for readability it is nicer to keep it as a separate function,
  also since it avoids inflating/rotationg smoothed states (for the EnKS).
  """
  do_infl = infl!=1.0

  if do_infl or rot:
    A, mu = anom(E)
    N,m   = E.shape
    T     = eye(N)

    if do_infl:
      T = infl * T

    if rot:
      T = genOG_1(N,rot) @ T

    E = mu + T@A
  return E



def add_noise(E, dt, noise, config):
  """
  Treatment of additive noise for ensembles.
  Settings for reproducing literature benchmarks may be found in
  mods/LA/raanes2015.py

  Ref: Raanes, Patrick Nima, Alberto Carrassi, and Laurent Bertino (2015):
  "Extending the square root method to account for additive forecast noise in ensemble methods."
  """
  method = config.get('fnoise_treatm','Stoch')

  if noise.C is 0: return E

  N,m  = E.shape
  A,mu = anom(E)
  Q12  = noise.C.Left
  Q    = noise.C.full

  def sqrt_core():
    T    = np.nan # cause error if used
    Qa12 = np.nan # cause error if used
    A2   = A.copy() # Instead of using (the implicitly nonlocal) A,
    # which changes A outside as well. NB: This is a bug in Datum!
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




@DA_Config
def SL_EAKF(loc_rad,N,taper='GC',ordr='rand',infl=1.0,rot=False,**kwargs):
  """
  Serial, covariance-localized EAKF.

  Ref: Karspeck, Alicia R., and Jeffrey L. Anderson. (2007):
  "Experimental implementation of an ensemble adjustment filter..."

  Used without localization, this should be equivalent
  (full ensemble equality) to the EnKF 'Serial'.
  See DAPPER/Misc/batch_vs_serial.py for some details.
  """
  def assimilator(stats,twin,xx,yy):
    f,h,chrono,X0 = twin.f, twin.h, twin.t, twin.X0

    n = N-1

    R    = h.noise
    Rm12 = h.noise.C.sym_sqrt_inv

    E = X0.sample(N)
    stats.assess(0,E=E)

    for k,kObs,t,dt in progbar(chrono.forecast_range):
      E = f(E,t-dt,dt)
      E = add_noise(E, dt, f.noise, kwargs)

      if kObs is not None:
        stats.assess(k,kObs,'f',E=E)
        y    = yy[kObs]
        inds = serial_inds(ordr, y, R, anom(E)[0])
            
        locf_at = h.loc_f(loc_rad, 'y2x', t, taper)
        for i,j in enumerate(inds):
          hE = h(E,t)
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

        E = post_process(E,infl,rot)

      stats.assess(k,kObs,E=E)
  return assimilator


def serial_inds(upd_a, y, cvR, A):
  if 'mono' in upd_a:
    # Not robust?
    inds = arange(len(y))
  elif 'sorted' in upd_a:
    dC = cvR.diag
    if np.all(dC == dC[0]):
      # Sort y by P
      dC = np.sum(A*A,0)/(N-1)
    inds = np.argsort(dC)
  else: # Default: random ordering
    inds = np.random.permutation(len(y))
  return inds
  


@DA_Config
def LETKF(loc_rad,N,taper='GC',approx=False,infl=1.0,rot=False,**kwargs):
  """
  Same as EnKF (sqrt), but with localization.

  Settings for reproducing literature benchmarks may be found in
  mods/Lorenz95/sak08.py

  Ref: Hunt, Brian R., Eric J. Kostelich, and Istvan Szunyogh. (2007):
  "Efficient data assimilation for spatiotemporal chaos..."
  """
  def assimilator(stats,twin,xx,yy):
    f,h,chrono,X0 = twin.f, twin.h, twin.t, twin.X0
    Rm12 = h.noise.C.sym_sqrt_inv

    E = X0.sample(N)
    stats.assess(0,E=E)

    for k,kObs,t,dt in progbar(chrono.forecast_range):
      E = f(E,t-dt,dt)
      E = add_noise(E, dt, f.noise, kwargs)

      if kObs is not None:
        stats.assess(k,kObs,'f',E=E)
        mu = mean(E,0)
        A  = E - mu

        hE = h(E,t)
        hx = mean(hE,0)
        YR = (hE-hx)  @ Rm12.T
        yR = (yy[kObs] - hx) @ Rm12.T

        locf_at = h.loc_f(loc_rad, 'x2y', t, taper)
        for i in range(f.m):
          # Localize
          local, coeffs = locf_at(i)
          if len(local) == 0: continue
          iY  = YR[:,local] * sqrt(coeffs)
          idy = yR[local]   * sqrt(coeffs)

          # Do analysis
          if approx:
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
          else:
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

        E = post_process(E,infl,rot)

        if 'sd' in locals():
          stats.trHK[kObs] = (sd**(-1.0) * sd**2).sum()/h.noise.m
        #else:
          # nevermind

      stats.assess(k,kObs,E=E)
  return assimilator


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
def myNewton(fun,deriv,x0,conf=1.0,xtol=1e-4,ytol=1e-4,itermax=10**4):
  "Simple implementation of Newton root-finding"
  itr, dx, Jx = 0, np.inf, fun(x0)
  while ytol<abs(Jx) and xtol<abs(dx) and itr<itermax:
    Dx  = deriv(x0)
    dx  = Jx/Dx * conf
    x0 -= dx
    Jx  = fun(x0)
  return x0

@DA_Config
def EnKF_N(N,infl=1.0,rot=False,Hess=False,**kwargs):
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
  def assimilator(stats,twin,xx,yy):
    # Unpack
    f,h,chrono,X0  = twin.f, twin.h, twin.t, twin.X0

    Rm12 = h.noise.C.sym_sqrt_inv
    Ri   = h.noise.C.inv

    # EnKF-N constants
    g    = 1             # Nullity of Y (obs anom's).
    #g   = max(1,N-h.m)  # TODO: No good
    eN   = (N+1)/N       # Effect of unknown mean
    clog = (N+g)/(N-1)   # Coeff in front of log term
    mode = eN/clog       # Mode of prior for lambda
    LowB = sqrt(mode)    # Lower bound for lambda^1

    E = X0.sample(N)
    stats.assess(0,E=E)

    for k,kObs,t,dt in progbar(chrono.forecast_range):
      E = f(E,t-dt,dt)
      E = add_noise(E, dt, f.noise, kwargs)

      if kObs is not None:
        stats.assess(k,kObs,'f',E=E)
        hE = h(E,t)
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
        # Derivatives
        dJ1  = lambda l: -2*l   * np.sum(pad0(s**2, m_Nm) * du**2/dgn(l)**2) \
               + -2*(1/fctr)*eN/l**3 \
               +  2*fctr*clog  /l
        dJ2  = lambda l: 8*l**2 * np.sum(pad0(s**4, m_Nm) * du**2/dgn(l)**3) \
               +  6*(1/fctr)*eN/l**4 \
               + -2*fctr*clog  /l**2
        # Find inflation factor
        l1 = myNewton(dJ1,dJ2, 1.0)
        #l1 = sp.optimize.fmin_bfgs(J, x0=[1], gtol=1e-4, disp=0)
        #l1 = sp.optimize.minimize_scalar(J, bracket=(LowB, 1e2), tol=1e-4).x

        # Turns EnKF-N into ETKF:
        #l1 = 1.0

        # Inflate prior.
        A *= l1
        Y *= l1

        # Compute ETKF (sym sqrt) update
        dgn     = lambda l: pad0( (l*s)**2, N ) + (N-1)
        Pw      = (V * dgn(l1)**(-1.0)) @ V.T
        w       = dy@Ri@Y.T@Pw

        if Hess:
          zeta  = (N-1)/l1**2
          Hw    = Y@Ri@Y.T + zeta*eye(N) - 2*zeta**2/(N+g)*np.outer(w,w)
          T     = funm_psd(Hw, lambda x: x**-.5) * sqrt(N-1) # sqrtm Woodbury?
        else:
          T     = (V * dgn(l1)**(-0.5)) @ V.T * sqrt(N-1)
          
        E = mu + w@A + T@A
        E = post_process(E,infl,rot)

        stats.infl[kObs] = l1
        stats.trHK[kObs] = (((l1*s)**2 + (N-1))**(-1.0)*s**2).sum()/h.noise.m

      stats.assess(k,kObs,E=E)
  return assimilator



@DA_Config
def EnKF_AdInf(N,infl=1.0,rot=False,Sb2=1.0,Nb=4,Fb=0.99,br=1.0,**kwargs):
  """
  """
  def assimilator(stats,twin,xx,yy):
    # Unpack
    f,h,chrono,X0  = twin.f, twin.h, twin.t, twin.X0

    nonlocal Nb, Sb2
    stats.aa = zeros(chrono.KObs+1)
    stats.bb = zeros(chrono.KObs+1)
    stats.Nb = zeros(chrono.KObs+1)

    Rm12 = h.noise.C.sym_sqrt_inv
    Ri   = h.noise.C.inv

    # EnKF-N constants
    g    = 1             # Nullity of Y (obs anom's).
    eN   = (N+1)/N       # Effect of unknown mean
    clog = (N+g)/(N-1)   # Coeff in front of log term
    mode = eN/clog       # Mode of prior for lambda
    LowB = sqrt(mode)    # Lower bound for lambda^1

    E = X0.sample(N)
    stats.assess(0,E=E)

    for k,kObs,t,dt in progbar(chrono.forecast_range):
      E = f(E,t-dt,dt)
      E = add_noise(E, dt, f.noise, kwargs)

      Nb = Nb*Fb # NB
      Sb2 -= (Sb2-1)*br # NB

      if kObs is not None:
        stats.assess(k,kObs,'f',E=E)
        hE = h(E,t)
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
        rb   = Nb/(N-1)
        fctr = sqrt(mode**(1/(1+PR)))
        J    = lambda ab:          np.sum(du**2/dgn(ab[0]*ab[1])) \
               + (1/fctr)*eN/ab[0]**2 \
               + fctr*clog*log(ab[0]**2) \
               + Sb2*rb/ab[1]**2 \
               + rb*log(ab[1]**2)
        # Find inflation factors
        a, b = sp.optimize.fmin_bfgs(J, x0=[1,sqrt(Sb2)], gtol=1e-4, disp=0)

        # TODO: Update Nb, Sb2
        Sb2 = (Nb*Sb2 + b**2)/(Nb + 1)
        Nb += 1

        # Aggregate factor
        l1 = a*b
        # Turns it into ETKF:
        #l1 = 1.0

        stats.aa[kObs] = a
        stats.bb[kObs] = b
        stats.Nb[kObs] = Nb

        # Inflate prior.
        A *= l1
        Y *= l1

        # Compute ETKF (sym sqrt) update
        dgn     = lambda l: pad0( (l*s)**2, N ) + (N-1)
        Pw      = (V * dgn(l1)**(-1.0)) @ V.T
        w       = dy@Ri@Y.T@Pw
        T       = (V * dgn(l1)**(-0.5)) @ V.T * sqrt(N-1)
          
        E = mu + w@A + T@A
        E = post_process(E,infl,rot)

        stats.infl[kObs] = l1
        stats.trHK[kObs] = (((l1*s)**2 + (N-1))**(-1.0)*s**2).sum()/h.noise.m

      stats.assess(k,kObs,E=E)
  return assimilator



@DA_Config
def iEnKF(upd_a,N,iMax=10,infl=1.0,rot=False,**kwargs):
  """
  Iterative EnKS.

  Loosely adapted from Bocquet ienks code and 
  Ref:.Bocquet, Marc, and Pavel Sakov. (2014):
  "An iterative ensemble Kalman smoother."

  Settings for reproducing literature benchmarks may be found in
  mods/Lorenz95/sak08.py
  mods/Lorenz95/sak12.py
  """
  def assimilator(stats,twin,xx,yy):
    f,h,chrono,X0,R = twin.f, twin.h, twin.t, twin.X0, twin.h.noise.C

    E = X0.sample(N)
    stats.iters = np.full(chrono.KObs+1,nan)
    stats.assess(0,E=E)

    for kObs in progbar(range(chrono.KObs+1)):
      xb0 = mean(E,0)
      A0  = E - xb0
      # Init
      w      = zeros(N)
      Tinv   = eye(N)
      T      = eye(N)
      for iteration in range(iMax):
        E = xb0 + w @ A0 + T @ A0
        for k,t,dt in chrono.DAW_range(kObs):
          E = f(E,t-dt,dt)
          E = add_noise(E, dt, f.noise, kwargs)
        if iteration==0:
          stats.assess(k,kObs,'f',E=E)

        hE = h(E,t)
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
      E = post_process(E,infl,rot)

      for k,t,dt in chrono.DAW_range(kObs):
        E = f(E,t-dt,dt)
        E = add_noise(E, dt, f.noise, kwargs)
        stats.assess(k,None,'u',E=E)
      stats.assess(k,kObs,'a',E=E)

      # TODO: It would be beneficial to do another (prior-regularized)
      # analysis at the end, after forecasting the E0 analysis.
  return assimilator


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



@DA_Config
def BnKF(N,infl=1.0,rot=False,**kwargs):
  import scipy.optimize as opt
  def assimilator(stats,twin,xx,yy):
    # Test settings:
    #from mods.Lorenz95.sak08 import setup
    #cfgs += EnKF_N(N=28)
    #cfgs += EnKF('PertObs',N=39,infl=1.06)
    #cfgs += BnKF(N=39,infl=1.03)

    # Unpack
    f,h,chrono,X0  = twin.f, twin.h, twin.t, twin.X0

    Rm12 = h.noise.C.sym_sqrt_inv
    Ri   = h.noise.C.inv

    # constants
    nu = N-1

    invm = lambda x: funm_psd(x, np.reciprocal)
    IN  = eye(N)
    Pi1 = np.outer(ones(N),ones(N))/N
    PiC = IN - Pi1

    # Init
    bb = N*ones(N)
    E  = X0.sample(N)
    stats.assess(0,E=E)

    for k,kObs,t,dt in progbar(chrono.forecast_range):
      E = f(E,t-dt,dt)
      E = add_noise(E, dt, f.noise, kwargs)

      if kObs is not None:
        stats.assess(k,kObs,'f',E=E)
        hE = h(E,t)
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

        E = post_process(E,infl,rot)
      stats.assess(k,kObs,E=E,w=1/bb) # TODO
  return assimilator




@DA_Config
def PartFilt(N,NER=1.0,resampl='Sys',reg=0,nuj=True,qroot=1.0,wroot=1.0,**kwargs):
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
       If resampling with some variant of 'Multinomial',
       no systematic bias is introduced.
   - qroot: "Inflate" (anneal) the proposal noise kernels
       by this root to increase diversity.
       The weights are updated to maintain un-biased-ness.
       Ref: [3], section VI-M.2

  [3]: Zhe Chen, 2003:
    "Bayesian Filtering: From Kalman Filters to Particle Filters, and Beyond"

  Settings for reproducing literature benchmarks may be found in
  mods/Lorenz95/boc10.py and mods/Lorenz95/boc10_m40.py.
  Other interesting settings include: mods/Lorenz63/sak12.py
  """
  # TODO:
  #if miN < 1:
    #miN = N*miN

  def assimilator(stats,twin,xx,yy):
    f,h,chrono,X0 = twin.f, twin.h, twin.t, twin.X0
    m, Rm12       = f.m, h.noise.C.sym_sqrt_inv

    E = X0.sample(N)
    w = 1/N*ones(N)

    stats.N_eff  = np.full(chrono.KObs+1,nan)
    stats.resmpl = zeros(chrono.KObs+1,dtype=bool)
    stats.innovs = np.full((chrono.KObs+1,N,h.m),nan)
    stats.assess(0,E=E,w=1/N)

    for k,kObs,t,dt in progbar(chrono.forecast_range):
      E = f(E,t-dt,dt)
      if f.noise.C is not 0:
        D  = randn((N,m))
        E += sqrt(dt*qroot)*(D@f.noise.C.Right)

        if qroot != 1.0:
          # Evaluate p/q (for each col of D) when q:=p**(1/qroot).
          w *= exp(-0.5*np.sum(D**2, axis=1) * (1 - 1/qroot))
          w /= w.sum()

      if kObs is not None:
        stats.assess(k,kObs,'f',E=E,w=w)

        innovs = (yy[kObs] - h(E,t)) @ Rm12.T
        w      = reweight(w,uni_innovs=innovs)

        stats.assess(k,kObs,'a',E=E,w=w)
        if trigger_resampling(w,NER,stats,kObs):
          C12    = reg*bandw(N,m)*raw_C12(E,w)
          #C12  *= sqrt(rroot) # Re-include?
          idx,w  = resample(w, resampl, wroot=wroot)
          E,chi2 = regularize(C12,E,idx,nuj)
          #if rroot != 1.0:
            # Compensate for rroot
            #w *= exp(-0.5*chi2*(1 - 1/rroot))
            #w /= w.sum()
      stats.assess(k,kObs,'u',E=E,w=w)
  return assimilator



@DA_Config
def OptPF(N,Qs,NER=1.0,resampl='Sys',reg=0,nuj=True,wroot=1.0,**kwargs):
  """
  "Optimal proposal" particle filter.
  OR
  Implicit particle filter.

  Note: Regularization (Qs) is here added BEFORE Bayes' rule.
  If Qs==0: OptPF should be equal to the bootsrap filter: PartFilt().

  Ref: Bocquet et al. (2010):
    "Beyond Gaussian statistical modeling in geophysical data assimilation"

  Settings for reproducing literature benchmarks may be found in
  mods/Lorenz95/boc10.py and mods/Lorenz95/boc10_m40.py.
  Other interesting settings include: mods/Lorenz63/sak12.py
  """
  def assimilator(stats,twin,xx,yy):
    f,h,chrono,X0 = twin.f, twin.h, twin.t, twin.X0
    m, R          = f.m, h.noise.C.full

    E = X0.sample(N)
    w = 1/N*ones(N)

    stats.N_eff  = np.full(chrono.KObs+1,nan)
    stats.resmpl = zeros(chrono.KObs+1,dtype=bool)
    stats.assess(0,E=E,w=1/N)

    for k,kObs,t,dt in progbar(chrono.forecast_range):
      E = f(E,t-dt,dt)
      if f.noise.C is not 0:
        E += sqrt(dt)*(randn((N,m))@f.noise.C.Right)

      if kObs is not None:
        stats.assess(k,kObs,'f',E=E,w=w)
        y = yy[kObs]

        hE = h(E,t)
        innovs = y - hE

        # EnKF-ish update
        s   = Qs*bandw(N,m)
        As  = s*raw_C12(E,w)
        Ys  = s*raw_C12(hE,w)
        C   = Ys.T@Ys + R
        KG  = As.T@mrdiv(Ys,C)
        E  += sample_quickly_with(As)[0]
        D   = h.noise.sample(N)
        dE  = KG @ (y-h(E,t)+D).T
        E   = E + dE.T

        # Importance weighting
        chi2   = innovs*mldiv(C,innovs.T).T
        logL   = -0.5 * np.sum(chi2, axis=1)
        w      = reweight(w,logL=logL)
        
        # Resampling
        stats.assess(k,kObs,'a',E=E,w=w)
        if trigger_resampling(w,NER,stats,kObs):
          C12    = reg*bandw(N,m)*raw_C12(E,w)
          idx,w  = resample(w, resampl, wroot=wroot)
          E,_    = regularize(C12,E,idx,nuj)

      stats.assess(k,kObs,'u',E=E,w=w)
  return assimilator

@DA_Config
def PFxN_EnKF(N,Qs,xN,re_use=True,NER=1.0,resampl='Sys',wroot_max=5,**kwargs):
  """
  Particle filter with EnKF-based proposal, q.
  Also employs xN duplication, as in PFxN.

  Recall that the proposals:
  Opt.: q_n(x) = c_n·N(x|x_n,Q     )·N(y|Hx,R)  (1)
  EnKF: q_n(x) = c_n·N(x|x_n,bar{B})·N(y|Hx,R)  (2)
  with c_n = p(y|x^{k-1}_n) being the composite proposal-analysis weight,
  and with Q possibly from regularization (rather than actual model noise).

  Here, we will use the posterior mean of (2) and cov of (1).
  Or maybe we should use x_a^n distributed according to a sqrt update?
  """
  def assimilator(stats,twin,xx,yy):
    f,h,chrono,X0 = twin.f, twin.h, twin.t, twin.X0
    m, Rm12, Ri   = f.m, h.noise.C.sym_sqrt_inv, h.noise.C.inv

    E = X0.sample(N)
    w = 1/N*ones(N)

    DD = None
    
    stats.N_eff  = np.full(chrono.KObs+1,nan)
    stats.wroot  = np.full(chrono.KObs+1,nan)
    stats.resmpl = zeros(chrono.KObs+1,dtype=bool)
    stats.assess(0,E=E,w=1/N)

    for k,kObs,t,dt in progbar(chrono.forecast_range):
      E = f(E,t-dt,dt)
      if f.noise.C is not 0:
        E += sqrt(dt)*(randn((N,m))@f.noise.C.Right)

      if kObs is not None:
        stats.assess(k,kObs,'f',E=E,w=w)
        y  = yy[kObs]
        hE = h(E,t)
        wD = w.copy()

        # Importance weighting
        innovs = (y - hE) @ Rm12.T
        w      = reweight(w,uni_innovs=innovs)
        
        # Resampling
        stats.assess(k,kObs,'a',E=E,w=w)
        if trigger_resampling(w,NER,stats,kObs):
          # Weighted covariance factors
          Aw = raw_C12(E,wD)
          Yw = raw_C12(hE,wD)

          # EnKF-without-pertubations update
          if N>m:
            C       = Yw.T @ Yw + h.noise.C.full
            KG      = mrdiv(Aw.T@Yw,C)
            cntrs   = E + (y-hE)@KG.T
            Pa      = Aw.T@Aw - KG@Yw.T@Aw
            P_cholU = funm_psd(Pa, sqrt)
            if DD is None or not re_use:
              DD    = randn((N*xN,m))
              chi2  = np.sum(DD**2, axis=1) * m/N
              log_q = -0.5 * chi2
          else:
            V,sig,UT = svd0( Yw @ Rm12.T )
            dgn      = pad0( sig**2, N ) + 1
            Pw       = (V * dgn**(-1.0)) @ V.T
            cntrs    = E + (y-hE)@Ri@Yw.T@Pw@Aw
            P_cholU  = (V*dgn**(-0.5)).T @ Aw
            # Generate N·xN random numbers from NormDist(0,1), and compute
            # log(q(x))
            if DD is None or not re_use:
              rnk   = min(m,N-1)
              DD    = randn((N*xN,N))
              chi2  = np.sum(DD**2, axis=1) * rnk/N
              log_q = -0.5 * chi2
            #NB: the DoF_linalg/DoF_stoch correction is only correct "on average".
            # It is inexact "in proportion" to V@V.T-Id, where V,s,UT = tsvd(Aw).
            # Anyways, we're computing the tsvd of Aw below, so might as well
            # compute q(x) instead of q(xi).

          # Duplicate
          ED  = cntrs.repeat(xN,0)
          wD  = wD.repeat(xN) / xN

          # Sample q
          AD = DD@P_cholU
          ED = ED + AD

          # log(prior_kernel(x))
          s         = Qs*bandw(N,m)
          innovs_pf = AD @ tinv(s*Aw)
          # NB: Correct: innovs_pf = (ED-E_orig) @ tinv(s*Aw)
          #     But it seems to make no difference on well-tuned performance !
          log_pf    = -0.5 * np.sum(innovs_pf**2, axis=1)

          # log(likelihood(x))
          innovs = (y - h(ED,t)) @ Rm12.T
          log_L  = -0.5 * np.sum(innovs**2, axis=1)

          # Update weights
          log_tot = log_L + log_pf - log_q
          wD      = reweight(wD,logL=log_tot)

          # Resample and reduce
          wroot = 1.0
          while wroot < wroot_max:
            idx,w  = resample(wD, resampl, wroot=wroot, N=N)
            dups   = sum(mask_unique_of_sorted(idx))
            if dups == 0:
              E = ED[idx]
              break
            else:
              wroot += 0.1
      stats.assess(k,kObs,'u',E=E,w=w)
  return assimilator




@DA_Config
def PFxN(N,Qs,xN,re_use=True,NER=1.0,resampl='Sys',wroot_max=5,**kwargs):
  """
  Idea: sample xN duplicates from each of the N kernels.
  Let resampling reduce it to N.
  Additional idea: employ w-adjustment to obtain N unique particles, without jittering.
  """
  def assimilator(stats,twin,xx,yy):
    f,h,chrono,X0 = twin.f, twin.h, twin.t, twin.X0
    m, Rm12       = f.m, h.noise.C.sym_sqrt_inv

    DD = None
    E  = X0.sample(N)
    w  = 1/N*ones(N)

    stats.N_eff  = np.full(chrono.KObs+1,nan)
    stats.resmpl = zeros(chrono.KObs+1,dtype=bool)
    stats.assess(0,E=E,w=1/N)

    for k,kObs,t,dt in progbar(chrono.forecast_range):
      E = f(E,t-dt,dt)
      if f.noise.C is not 0:
        E += sqrt(dt)*(randn((N,m))@f.noise.C.Right)

      if kObs is not None:
        stats.assess(k,kObs,'f',E=E,w=w)
        y  = yy[kObs]
        wD = w.copy()

        innovs = (y - h(E,t)) @ Rm12.T
        w      = reweight(w,uni_innovs=innovs)

        stats.assess(k,kObs,'a',E=E,w=w)
        if trigger_resampling(w,NER,stats,kObs):
          # Compute kernel colouring matrix
          cholR = Qs*bandw(N,m)*raw_C12(E,wD)
          cholR = chol_reduce(cholR)

          # Generate N·xN random numbers from NormDist(0,1)
          if DD is None or not re_use:
            DD = randn((N*xN,m))

          # Duplicate and jitter
          ED  = E.repeat(xN,0)
          wD  = wD.repeat(xN) / xN
          ED += DD[:,:len(cholR)]@cholR

          # Update weights
          innovs = (y - h(ED,t)) @ Rm12.T
          wD     = reweight(wD,uni_innovs=innovs)

          # Resample and reduce
          wroot = 1.0
          while wroot < wroot_max:
            idx,w = resample(wD, resampl, wroot=wroot, N=N)
            dups  = sum(mask_unique_of_sorted(idx))
            if dups == 0:
              E = ED[idx]
              break
            else:
              wroot += 0.1
      stats.assess(k,kObs,'u',E=E,w=w)
  return assimilator



def trigger_resampling(w,NER,stats,kObs):
  "Return boolean: N_effective <= threshold. Also write stats."
  N_eff              = 1/(w@w)
  do_resample        = N_eff <= len(w)*NER
  stats.N_eff[kObs]  = N_eff
  stats.resmpl[kObs] = do_resample
  return do_resample

def reweight(w,lklhd=None,logL=None,uni_innovs=None):
  """
  Do Bayes' rule (for the empirical distribution of an importance sample).
  If input is 'uni_innovs', the likelihood is assumed Gaussian.
  Do computations in log-space, for at least 2 reasons:
  - Normalization: will fail if sum==0 (if all innov's are large).
  - Num. precision: lklhd*w should have better precision in log space.
  Output is non-log, for the purpose of assessment and resampling.
  """
  assert all_but_1_is_None(lklhd,logL,uni_innovs), \
      "Input error. Only specify one of lklhd, logL, uni_innovs"

  # Get log-values.
  # Use context manager 'errstate' to not warn for log(0) = -inf.
  # Note: the case when all(w==0) will cause nan's,
  #       which should cause errors outside.
  with np.errstate(divide='ignore'):
    logw = log(w)        
    if lklhd is not None:
      logL = log(lklhd)
    elif uni_innovs is not None:
      chi2 = np.sum(uni_innovs**2, axis=1)
      logL = -0.5 * chi2

  logw   = logw + logL   # Bayes' rule in log-space
  logw  -= logw.max()    # Avoid numerical error
  w      = exp(logw)     # non-log
  w     /= w.sum()       # normalize
  return w

def raw_C12(E,w):
  """
  Compute the 'raw' matrix-square-root of the ensemble' covariance.
  The weights are used both for the mean and anomalies (raw sqrt).

  Note: anomalies (and thus cov) are weighted,
  and also computed based on a weighted mean.
  """
  # If weights are degenerate: use unweighted covariance to avoid C=0.
  if weight_degeneracy(w):
    w = ones(len(w))/len(w)
    # PS: 'avoid_pathological' already treated here.

  mu  = w@E
  A   = E - mu
  ub  = unbias_var(w, avoid_pathological=False)
  C12 = sqrt(ub*w[:,None]) * A
  return C12

def mask_unique_of_sorted(idx):
  "NB: returns a mask which is True at [i] iff idx[i] is NOT unique."
  duplicates  = idx==np.roll(idx,1)
  duplicates |= idx==np.roll(idx,-1)
  return duplicates

def bandw(N,m):
  """"
  Optimal bandwidth (not bandwidth^2), as per Scott's rule-of-thumb.
  Refs: [1] section 12.2.2, and [2] #Rule_of_thumb
  [1]: Doucet, de Freitas, Gordon, 2001:
    "Sequential Monte Carlo Methods in Practice"
  [2] wikipedia.org/wiki/Multivariate_kernel_density_estimation
  """
  return N**(-1/(m+4))


def regularize(C12,E,idx,no_uniq_jitter):
  """
  After resampling some of the particles will be identical.
  Therefore, if noise.is_deterministic: some noise must be added.
  This is adjusted by the regularization 'reg' factor
  (so-named because Dirac-deltas are approximated  Gaussian kernels),
  which controls the strength of the jitter.
  This causes a bias. But, as N-->∞, the reg. bandwidth-->0, i.e. bias-->0.
  Ref: [1], section 12.2.2.

  [1]: Doucet, de Freitas, Gordon, 2001:
    "Sequential Monte Carlo Methods in Practice"
  """
  # Select
  E = E[idx]

  # Jitter
  if no_uniq_jitter:
    dups         = mask_unique_of_sorted(idx)
    sample, chi2 = sample_quickly_with(C12, N=sum(dups))
    E[dups]     += sample
  else:
    sample, chi2 = sample_quickly_with(C12, N=len(E))
    E           += sample

  return E, chi2


def resample(w,kind='Systematic',N=None,wroot=1.0):
  """
  Multinomial resampling.

  - kind: 'Systematic', 'Residual' or 'Stochastic'.
    'Stochastic' corresponds to np.random.choice() or np.random.multinomial().
    'Systematic' and 'Residual' are more systematic (less stochastic)
    varaitions of 'Stochastic' sampling.
    Among the three, 'Systematic' is fastest, introduces the least noise,
    and brings continuity benefits for localized particle filters,
    and is therefore generally prefered.
    Example: see docs/test_resample.py.

  - N can be different from len(w)
    (e.g. in case some particles have been elimintated).

  - wroot: Adjust weights before resampling by this root to
    promote particle diversity and mitigate thinning.
    The outcomes of the resampling are then weighted to maintain un-biased-ness.
    Ref: [3], section 3.1

  Note: (a) resampling methods are beneficial because they discard
  low-weight ("doomed") particles and reduce the variance of the weights.
  However, (b) even unbiased/rigorous resampling methods introduce noise;
  (increases the var of any empirical estimator, see [1], section 3.4).
  How to unify the seemingly contrary statements of (a) and (b) ?
  By recognizing that we're in the *sequential/dynamical* setting,
  and that *future* variance may be expected to be lower by focusing
  on the high-weight particles which we anticipate will 
  have more informative (and less variable) future likelihoods.

  [1]: Doucet, Johansen, 2009, v1.1:
    "A Tutorial on Particle Filtering and Smoothing: Fifteen years later." 
  [2]: Van Leeuwen, 2009: "Particle Filtering in Geophysical Systems"
  [3]: Liu, Chen Longvinenko, 2001:
    "A theoretical framework for sequential importance sampling with resampling"
  """

  assert(abs(w.sum()-1) < 1e-5)

  # Input parsing
  N_o = len(w)  # N _original
  if N is None: # N to sample
    N = N_o

  # Compute factors s such that s*w := w**(1/wroot). 
  if wroot!=1.0:
    s   = ( w**(1/wroot - 1) ).clip(max=1e100)
    s  /= (s*w).sum()
    sw  = s*w
  else:
    s   = ones(N_o)
    sw  = w

  # Do the actual resampling
  idx = _resample(sw,kind,N_o,N)

  w  = 1/s[idx] # compensate for above scaling by s
  w /= w.sum()  # normalize

  return idx, w


def _resample(w,kind,N_o,N):
  "Core functionality for resample(). See its docstring."
  if kind in ['Stochastic','Stoch']:
    # van Leeuwen [2] also calls this "probabilistic" resampling
    idx = np.random.choice(N_o,N,replace=True,p=w)
    # np.random.multinomial is faster (slightly different usage) ?
  elif kind in ['Residual','Res']:
    # Doucet [1] also calls this "stratified" resampling.
    w_N   = w*N             # upscale
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
  elif kind in ['Systematic','Sys']:
    # van Leeuwen [2] also calls this "stochastic universal" resampling
    U     = rand(1) / N
    CDF_a = U + arange(N)/N
    CDF_o = np.cumsum(w)
    #idx  = CDF_a <= CDF_o[:,None]
    #idx  = np.argmax(idx,axis=0) # Finds 1st. SO/a/16244044/
    idx   = np.searchsorted(CDF_o,CDF_a)
  else:
    raise KeyError
  return idx

def sample_quickly_with(C12,N=None):
  """
  Gaussian sampling in the quickest fashion,
  which depends on the size of the colouring matrix 'C12'.
  """
  (N_,m) = C12.shape
  if N is None: N = N_
  if N_ > 2*m:
    cholR  = chol_reduce(C12)
    D      = randn((N,cholR.shape[0]))
    chi2   = np.sum(D**2, axis=1)
    sample = D@cholR
  else:
    chi2_compensate_for_rank = min(m/N_,1.0)
    D      = randn((N,N_))
    chi2   = np.sum(D**2, axis=1) * chi2_compensate_for_rank
    sample = D@C12
  return sample, chi2  





@DA_Config
def EnCheat(upd_a,N,infl=1.0,rot=False,**kwargs):
  """
  A baseline/reference method.
  Ensemble method that cheats: it knows the truth.
  Nevertheless, its error will not necessarily be 0,
  because the truth may be outside of the ensemble subspace.
  This method is just to provide a baseline for comparison with other methods.
  It may very well beat the particle filter with N=infinty.
  NB: The forecasts (and their rmse) are given by the standard EnKF.
  """
  def assimilator(stats,twin,xx,yy):
    f,h,chrono,X0 = twin.f, twin.h, twin.t, twin.X0

    E = X0.sample(N)
    stats.assess(0,E=E)

    for k,kObs,t,dt in progbar(chrono.forecast_range):
      E = f(E,t-dt,dt)
      E = add_noise(E, dt, f.noise, kwargs)

      if kObs is not None:
        # Standard EnKF analysis
        hE = h(E,t)
        y  = yy[kObs]
        E  = EnKF_analysis(E,hE,h.noise,y,upd_a,stats,kObs)
        E  = post_process(E,infl,rot)

        # Cheating (only used for stats)
        w,res,_,_ = sla.lstsq(E.T, xx[k])
        if not res.size:
          res = 0
        res = diag((res/twin.f.m) * ones(twin.f.m))
        opt = w @ E
        # NB: Center on the optimal solution?
        #E += opt - mean(E,0)

      stats.assess(k,kObs,mu=opt,Cov=res)
  return assimilator



@DA_Config
def Climatology(**kwargs):
  """
  A baseline/reference method.
  Note that the "climatology" is computed from truth, which might be
  (unfairly) advantageous if the simulation is too short (vs mixing time).
  """
  def assimilator(stats,twin,xx,yy):
    f,h,chrono,X0 = twin.f, twin.h, twin.t, twin.X0

    muC = mean(xx,0)
    AC  = xx - muC
    PC  = CovMat(AC,'A')

    stats.assess(0,mu=muC,Cov=PC)
    stats.trHK[:] = 0

    for k,kObs,_,_ in progbar(chrono.forecast_range):
      stats.assess(k,kObs,'fau',mu=muC,Cov=PC)
  return assimilator


@DA_Config
def OptInterp(**kwargs):
  """
  Optimal Interpolation -- a baseline/reference method.
  Uses the Kalman filter equations,
  but with a prior from the Climatology.
  """
  def assimilator(stats,twin,xx,yy):
    f,h,chrono,X0 = twin.f, twin.h, twin.t, twin.X0

    # Get H.
    msg  = "For speed, only time-independent H is supported."
    H    = h.jacob(np.nan, np.nan)
    if not np.all(np.isfinite(H)): raise AssimFailedError(msg)

    # Compute "climatological" Kalman gain
    muC = mean(xx,0)
    AC  = xx - muC
    PC  = (AC.T @ AC) / (xx.shape[0] - 1)
    KG  = mrdiv(PC@H.T, H@PC@H.T + h.noise.C.full)

    # Setup scalar "time-series" covariance dynamics.
    # ONLY USED FOR DIAGNOSTICS, not to change the Kalman gain.
    Pa    = (eye(f.m) - KG@H) @ PC
    CorrL = estimate_corr_length(AC.ravel(order='F'))
    WaveC = wave_crest(trace(Pa)/trace(2*PC),CorrL)

    # Init
    mu = muC
    stats.assess(0,mu=mu,Cov=PC)

    for k,kObs,t,dt in progbar(chrono.forecast_range):
      # Forecast
      mu = f(mu,t-dt,dt)
      if kObs is not None:
        stats.assess(k,kObs,'f',mu=muC,Cov=PC)
        # Analysis
        mu = muC + KG@(yy[kObs] - h(muC,t))
      stats.assess(k,kObs,mu=mu,Cov=2*PC*WaveC(k,kObs))
  return assimilator


@DA_Config
def Var3D(infl=1.0,**kwargs):
  """
  3D-Var -- a baseline/reference method.
  Uses the Kalman filter equations,
  but with a prior covariance estimated from the Climatology
  and a scalar time-series approximation to the dynamics
  (that does NOT use the innovation to estimate the backgroiund covariance).
  """
  # TODO: The wave-crest yields good results for sak08, but not for boc10 
  def assimilator(stats,twin,xx,yy):
    f,h,chrono,X0 = twin.f, twin.h, twin.t, twin.X0

    # Compute "climatology"
    muC = mean(xx,0)
    AC  = xx - muC
    PC  = (AC.T @ AC)/(xx.shape[0] - 1)

    # Setup scalar "time-series" covariance dynamics
    CorrL = estimate_corr_length(AC.ravel(order='F'))
    WaveC = wave_crest(0.5,CorrL) # Nevermind careless W0 init

    # Init
    mu = muC
    P  = PC
    stats.assess(0,mu=mu,Cov=P)

    for k,kObs,t,dt in progbar(chrono.forecast_range):
      # Forecast
      mu = f(mu,t-dt,dt)
      P  = 2*PC*WaveC(k)

      if kObs is not None:
        stats.assess(k,kObs,'f',mu=mu,Cov=P)
        # Analysis
        P *= infl
        H  = h.jacob(mu,t)
        KG = mrdiv(P@H.T, H@P@H.T + h.noise.C.full)
        KH = KG@H
        mu = mu + KG@(yy[kObs] - h(mu,t))

        # Re-calibrate wave_crest with new W0 = Pa/(2*PC).
        # Note: obs innovations are not used to estimate P!
        Pa    = (eye(f.m) - KH) @ P
        WaveC = wave_crest(trace(Pa)/trace(2*PC),CorrL)

      stats.assess(k,kObs,mu=mu,Cov=2*PC*WaveC(k,kObs))
  return assimilator



def wave_crest(W0,L):
  """
  Return a sigmoid [function W(k)] that may be used
  to provide scalar approximations to covariance dynamics. 

  We use the logistic function for the sigmoid.
  This has theoretical benefits: it's the solution of the
  "population growth" ODE: dE/dt = a*E*(1-E/E(∞)).
  PS: It might be better to use the "error growth ODE" of Lorenz/Dalcher/Kalnay,
  but this has a significantly more complicated closed-form solution,
  and reduces to the above ODE when there's no model error (ODE source term).

  As "any sigmoid", W is symmetric around 0 and W(-∞)=0 and W(∞)=1.
  It is further fitted such that
   - W has a scale (slope at 0) equal to that of f(t) = 1-exp(-t/L),
     where L is suppsed to be the system's decorrelation length (L),
     as detailed in doc/wave_crest.jpg.
   - W has a shift (translation) such that it takes the value W0 at k=0.
     Typically, W0 = 2*PC/Pa, where
     2*PC = 2*Var(free_run) = Var(mu-truth), and Pa = Var(analysis).

  The best way to illustrate W(k) and test it is to:
   - set dkObs very large, to see a long evolution;
   - set store_u = True, to store intermediate stats;
   - plot_time_series (var vs err).
  """
  sigmoid = lambda t: 1/(1+exp(-t))
  inv_sig = lambda s: log(s/(1-s))
  shift   = inv_sig(W0) # "reset" point
  scale   = 1/(2*L)     # derivative of exp(-t/L) at t=1/2

  def W(k,reset=False):
    # Manage intra-DAW counter, dk.
    if reset is None:
      dk = k - W.prev_obs
    elif reset > 0:
      W.prev_obs = k
      dk = k - W.prev_obs # = 0
    else:
      dk = k - W.prev_obs
    # Compute
    return sigmoid(shift + scale*dk)

  # Initialize W.prev_obs: provides persistent ref [for W(k)] to compute dk.
  W.prev_obs = (shift-inv_sig(0.5))/scale 

  return W


# TODO: Clean up
@DA_Config
def ExtRTS(infl=1.0,**kwargs):
  """
  """
  def assimilator(stats,twin,xx,yy):
    f,h,chrono,X0 = twin.f, twin.h, twin.t, twin.X0

    R  = h.noise.C.full
    Q  = 0 if f.noise.C==0 else f.noise.C.full

    mu    = zeros((chrono.K+1,f.m))
    P     = zeros((chrono.K+1,f.m,f.m))

    # Forecasted values
    muf   = zeros((chrono.K+1,f.m))
    Pf    = zeros((chrono.K+1,f.m,f.m))
    Ff    = zeros((chrono.K+1,f.m,f.m))

    mu[0] = X0.mu
    P [0] = X0.C.full

    stats.assess(0,mu=mu[0],Cov=P[0])

    # Forward pass
    for k,kObs,t,dt in progbar(chrono.forecast_range, 'ExtRTS->'):
      mu[k]  = f(mu[k-1],t-dt,dt)
      F      = f.jacob(mu[k-1],t-dt,dt) 
      P [k]  = infl**(dt)*(F@P[k-1]@F.T) + dt*Q

      # Store forecast and Jacobian
      muf[k] = mu[k]
      Pf [k] = P [k]
      Ff [k] = F

      if kObs is not None:
        stats.assess(k,kObs,'f',mu=mu[k],Cov=P[k])
        H     = h.jacob(mu[k],t)
        KG    = mrdiv(P[k] @ H.T, H@P[k]@H.T + R)
        y     = yy[kObs]
        mu[k] = mu[k] + KG@(y - h(mu[k],t))
        KH    = KG@H
        P[k]  = (eye(f.m) - KH) @ P[k]
        stats.assess(k,kObs,'a',mu=mu[k],Cov=P[k])

    # Backward pass
    for k in progbar(range(chrono.K)[::-1],'ExtRTS<-'):
      J     = mrdiv(P[k]@Ff[k+1].T, Pf[k+1])
      mu[k] = mu[k]  + J @ (mu[k+1]  - muf[k+1])
      P[k]  = P[k] + J @ (P[k+1] - Pf[k+1]) @ J.T
    for k in progbar(range(chrono.K+1),desc='Assess'):
      stats.assess(k,mu=mu[k],Cov=P[k])

  return assimilator


@DA_Config
def ExtKF(infl=1.0,**kwargs):
  """
  The extended Kalman filter.
  A baseline/reference method.

  If everything is linear-Gaussian, this provides the exact solution
  to the Bayesian filtering equations.

  - infl (inflation) may be specified.
    Default: 1.0 (i.e. none), as is optimal in the lin-Gauss case.
    Gets applied at each dt, with infl_per_dt := inlf**(dt), so that 
    infl_per_unit_time == infl.
    Specifying it this way (per unit time) means less tuning.
  """
  def assimilator(stats,twin,xx,yy):
    f,h,chrono,X0 = twin.f, twin.h, twin.t, twin.X0

    R  = h.noise.C.full
    Q  = 0 if f.noise.C==0 else f.noise.C.full

    mu = X0.mu
    P  = X0.C.full

    stats.assess(0,mu=mu,Cov=P)

    for k,kObs,t,dt in progbar(chrono.forecast_range):
      
      mu = f(mu,t-dt,dt)
      F  = f.jacob(mu,t-dt,dt) 
      P  = infl**(dt)*(F@P@F.T) + dt*Q

      # Of academic interest? Higher-order linearization:
      # mu_i += 0.5 * (Hessian[f_i] * P).sum()

      if kObs is not None:
        stats.assess(k,kObs,'f',mu=mu,Cov=P)
        H  = h.jacob(mu,t)
        KG = mrdiv(P @ H.T, H@P@H.T + R)
        y  = yy[kObs]
        mu = mu + KG@(y - h(mu,t))
        KH = KG@H
        P  = (eye(f.m) - KH) @ P

        stats.trHK[kObs] = trace(KH)/f.m

      stats.assess(k,kObs,mu=mu,Cov=P)
  return assimilator





@DA_Config
def LNETF(loc_rad,N,taper='GC',infl=1.0,Rs=1.0,rot=False,**kwargs):
  """
  The Nonlinear-Ensemble-Transform-Filter (localized).

  Ref: Julian Tödter and Bodo Ahrens (2014):
  "A Second-Order Exact Ensemble Square Root Filter for Nonlinear Data Assimilation"

  It is (supposedly) a deterministic upgrade of the NLEAF of Lei and Bickel (2011).

  Settings for reproducing literature benchmarks may be found in
  mods/Lorenz95/tod15.py
  """
  def assimilator(stats,twin,xx,yy):
    f,h,chrono,X0 = twin.f, twin.h, twin.t, twin.X0
    Rm12 = h.noise.C.sym_sqrt_inv

    E = X0.sample(N)
    stats.assess(0,E=E)

    for k,kObs,t,dt in progbar(chrono.forecast_range):
      E = f(E,t-dt,dt)
      E = add_noise(E, dt, f.noise, kwargs)

      if kObs is not None:
        stats.assess(k,kObs,'f',E=E)
        mu = mean(E,0)
        A  = E - mu

        hE = h(E,t)
        hx = mean(hE,0)
        YR = (hE-hx)  @ Rm12.T
        yR = (yy[kObs] - hx) @ Rm12.T

        locf_at = h.loc_f(loc_rad, 'x2y', t, taper)
        for i in range(f.m):
          # Localize
          local, coeffs = locf_at(i)
          if len(local) == 0: continue
          iY  = YR[:,local] * sqrt(coeffs)
          idy = yR[local]   * sqrt(coeffs)

          # NETF:
          # This "paragraph" is the only difference to the LETKF.
          innovs = (idy-iY)/Rs
          if 'laplace' in str(type(h.noise)).lower():
            w    = laplace_lklhd(innovs)
          else:
            w    = reweight(ones(N),uni_innovs=innovs)
          dmu    = w@A[:,i]
          AT     = sqrt(N)*funm_psd(diag(w) - np.outer(w,w), sqrt)@A[:,i]

          E[:,i] = mu[i] + dmu + AT
        E = post_process(E,infl,rot)
      stats.assess(k,kObs,E=E)
  return assimilator

def laplace_lklhd(xx):
  """
  Compute likelihood of xx wrt. the sampling distribution
  LaplaceParallelRV(C=I), i.e., for x in xx:
  p(x) = exp(-sqrt(2)*|x|_1) / sqrt(2).
  """
  logw   = -sqrt(2)*np.sum(np.abs(xx), axis=1)
  logw  -= logw.max()    # Avoid numerical error
  w      = exp(logw)     # non-log
  w     /= w.sum()       # normalize
  return w
