"""Variational methods (iEnKS, 4D-Var, etc)"""

from dapper import *

@DA_Config
def iEnKS(upd_a,N,Lag=1,nIter=10,wtol=0,MDA=False,step=False,bundle=False,xN=None,infl=1.0,rot=False,**kwargs):
  """Iterative EnKS.

  Special cases: EnRML, ES-MDA, iEnKF, EnKF [Raa19b]_.

  As in [Boc14]_, optimization uses Gauss-Newton. See [Boc12]_ for Levenberg-Marquardt.
  If MDA=True, then there's not really any optimization, but rather Gaussian annealing.

  Args:
    upd_a (str): 
      Analysis update form (flavour). One of:

      - "Sqrt"   : as in ETKF  , using a deterministic matrix square root transform.
      - "PertObs": as in EnRML , using stochastic, perturbed-observations.
      - "Order1" : as in DEnKF of [Sak08a]_.

    Lag   : 
      Length of the DA window (DAW), in multiples of dkObs (i.e. cycles).

      - Lag=1 (default) => iterative "filter" iEnKF [Sak12]_.
      - Lag=0           => maximum-likelihood filter [Zup05]_.

    Shift : How far (in cycles) to slide the DAW.
            Fixed at 1 for code simplicity.

    nIter : Maximal num. of iterations used (>=1).
            Supporting nIter==0 requires more code than it's worth.

    wtol  : Rel. tolerance defining convergence.
            Default: 0 => always do nIter iterations.
            Recommended: 1e-5.

    MDA   : Use iterations of the "multiple data assimlation" type.

    bundle: Use finite-diff. linearization instead of of least-squares regression.
            Makes the iEnKS very much alike the iterative, extended KF (IEKS).

    xN    : If set, use EnKF_N() pre-inflation. See further documentation there.

  Total number of model simulations (of duration dtObs): N * (nIter*Lag + 1).
  (due to boundary cases: only asymptotically valid)
  
  References: [Boc12]_, [Boc13]_, [Boc14]_,

  Settings for reproducing literature benchmarks may be found in

  - :mod:`dapper.mods.Lorenz95.sak08`
  - :mod:`dapper.mods.Lorenz63.sak12`
  - :mod:`dapper.mods.Lorenz63.boc12`
  """

  # NB It's very difficult to preview what should happen to all of the time indices in all cases of
  # nIter and Lag. => Any changes to this function must be unit-tested via scripts/test_iEnKS.py.

  # TODO:
  # - step length
  # - Implement quasi-static assimilation. Boc notes:
  #   * The 'balancing step' is complicated.
  #   * Trouble playing nice with '-N' inflation estimation.

  def assimilator(stats,HMM,xx,yy):
    Dyn,Obs,chrono,X0,R,KObs, N1 = HMM.Dyn, HMM.Obs, HMM.t, HMM.X0, HMM.Obs.noise.C, HMM.t.KObs, N-1
    Rm12 = R.sym_sqrt_inv

    assert Dyn.noise.C is 0, "Q>0 not yet supported. TODO: implement Sakov et al 2017"
    
    if bundle: EPS = 1e-4 # Sakov/Boc use T=EPS*eye(N), with EPS=1e-4, but I
    else     : EPS = 1.0  # prefer using  T=EPS*T, yielding a conditional shape of the bundle.

    # Initial ensemble
    E = X0.sample(N)

    # Loop over DA windows (DAW).
    for kObs in progbar(arange(-1,KObs+Lag+1)):
        kLag = kObs-Lag
        DAW = range( max(0, kLag+1), min(kObs, KObs) + 1)

        # Assimilation (if ∃ "not-fully-assimlated" obs).
        if 0 <= kObs <= KObs:  

            # Init iterations.
            X0,x0 = center(E) # Decompose ensemble.
            w     = zeros(N)  # Control vector for the mean state.
            T     = eye(N)    # Anomalies transform matrix.
            Tinv  = eye(N)    # Explicit Tinv [instead of tinv(T)] allows for merging MDA code
                              # with iEnKS/EnRML code, and flop savings in 'Sqrt' case.

            for iteration in arange(nIter):
                # Reconstruct smoothed ensemble.
                E = x0 + (w + EPS*T)@X0 
                # Forecast.
                for kCycle in DAW:
                  for k,t,dt in chrono.cycle(kCycle):
                    E = Dyn(E,t-dt,dt)
                # Observe.
                Eo = Obs(E,t)

                # Undo the bundle scaling of ensemble.
                if EPS!=1.0:
                  E  = inflate_ens(E ,1/EPS)
                  Eo = inflate_ens(Eo,1/EPS)

                # Assess forecast stats; store {Xf, T_old} for analysis assessment.
                if iteration==0:
                  stats.assess(k,kObs,'f',E=E)
                  Xf,xf = center(E)
                T_old = T

                # Prepare analysis.
                y      = yy[kObs]          # Get current obs.
                Y,xo   = center(Eo)        # Get obs {anomalies, mean}.
                dy     = (y - xo) @ Rm12.T # Transform obs space.
                Y      = Y        @ Rm12.T # Transform obs space.
                Y0     = Tinv @ Y          # "De-condition" the obs anomalies.
                V,s,UT = svd0(Y0)          # Decompose Y0.

                # Set "cov normlzt fctr" za ("effective ensemble size") => pre-infl^2 = (N-1)/za.
                if xN is None: za  = N1
                else         : za  = zeta_a(*hyperprior_coeffs(s,N,xN), w)
                if MDA       : za *= nIter # inflation (factor: nIter) of the ObsErrCov.

                # Post. cov (approx) of w, estimated at current iteration, raised to power.
                Cowp = lambda expo: (V * (pad0(s**2,N) + za)**-expo) @ V.T
                Cow1 = Cowp(1.0)

                if MDA: # View update as annealing (progressive assimilation).
                    Cow1 = Cow1 @ T # apply previous update
                    dw = dy @ Y.T @ Cow1
                    if 'PertObs' in upd_a:           #== "ES-MDA". By Emerick/Reynolds.
                      D     = mean0(randn(Y.shape)) * sqrt(nIter)
                      T    -= (Y + D) @ Y.T @ Cow1
                    elif 'Sqrt' in upd_a:            #== "ETKF-ish". By Raanes.
                      T     = Cowp(0.5) * sqrt(za) @ T
                    elif 'Order1' in upd_a:          #== "DEnKF-ish". By Emerick.
                      T    -= 0.5 * Y @ Y.T @ Cow1
                    # Tinv = eye(N) [as initialized] coz MDA does not de-condition.

                else: # View update as Gauss-Newton optimzt. of log-posterior.
                    grad  = Y0@dy - w*za             # Cost function gradient
                    dw    = grad@Cow1                # Gauss-Newton step
                    if 'Sqrt' in upd_a:              # =="ETKF-ish". By Bocquet/Sakov.
                      T     = Cowp(0.5) * sqrt(N1)   # Sqrt-transforms
                      Tinv  = Cowp(-.5) / sqrt(N1)   # Saves time [vs tinv(T)] when Nx<N
                    elif 'PertObs' in upd_a:         # =="EnRML". By Oliver/Chen/Raanes/Evensen/Stordal.
                      D     = mean0(randn(Y.shape)) if iteration==0 else D
                      gradT = -(Y+D)@Y0.T + N1*(eye(N) - T)
                      T     = T + gradT@Cow1
                      # Tinv= tinv(T, threshold=N1)  # unstable
                      Tinv  = inv(T+1)               # the +1 is for stability.
                    elif 'Order1' in upd_a:          #== "DEnKF-ish". By Raanes. 
                      # Included for completeness; does not make much sense.
                      gradT = -0.5*Y@Y0.T + N1*(eye(N) - T)
                      T     = T + gradT@Cow1
                      Tinv  = tinv(T, threshold=N1)

                w += dw
                if dw@dw < wtol*N:
                  break
            # END loop iteration

            # Assess (analysis) stats. The final_increment is a linearization to
            # (i) avoid re-running the model and (ii) reproduce EnKF in case nIter==1. 
            final_increment = (dw+T-T_old)@Xf # See dpr_data/doc_snippets/iEnKS_Ea.jpg.
            stats.assess(k,   kObs, 'a', E=E+final_increment)
            stats.iters      [kObs] = iteration+1
            if xN: stats.infl[kObs] = sqrt(N1/za)

            # Final (smoothed) estimate of E at [kLag].
            E = x0 + (w+T)@X0
            E = post_process(E,infl,rot)
        # END assimilation block

        # Slide/shift DAW by propagating smoothed ('s') ensemble from [kLag].
        if -1 <= kLag < KObs: 
          if kLag>=0: stats.assess(chrono.kkObs[kLag],kLag,'s',E=E)
          for k,t,dt in chrono.cycle(kLag+1):
            stats.assess(k-1,None,'u',E=E)
            E = Dyn(E,t-dt,dt)

    # END loop kObs
    stats.assess(k,KObs,'us',E=E)
  return assimilator

@DA_Config
def iLEnKS(upd_a,N,loc_rad,taper='GC',Lag=1,nIter=10,xN=1.0,infl=1.0,rot=False,**kwargs):
  """Iterative, Localized EnKS-N. [Boc16]_

  Based on iEnKS() and LETKF() codes,
  which describes the other input arguments.

  - upd_a : - 'Sqrt' (i.e. ETKF)
            - '-N' (i.e. EnKF-N)

  Settings for reproducing literature benchmarks may be found in
  :mod:`dapper.mods.Lorenz95.boc15loc`
  """

  def assimilator(stats,HMM,xx,yy):
    Dyn,Obs,chrono,X0,R,KObs, N1 = HMM.Dyn, HMM.Obs, HMM.t, HMM.X0, HMM.Obs.noise.C, HMM.t.KObs, N-1
    assert Dyn.noise.C is 0, "Q>0 not yet supported. See Sakov et al 2017: 'An iEnKF with mod. error'"

    # Init DA cycles
    E = X0.sample(N)

    # Loop DA cycles
    for kObs in progbar(arange(KObs+1)):
        # Set (shifting) DA Window. Shift is 1.
        DAW_0  = kObs-Lag+1
        DAW    = arange(max(0,DAW_0),DAW_0+Lag)
        DAW_dt = chrono.ttObs[DAW[-1]] - chrono.ttObs[DAW[0]] + chrono.dtObs

        # Get localization setup (at time t)
        state_batches, obs_taperer = Obs.localizer(loc_rad, 'x2y', chrono.ttObs[kObs], taper)
        nBatch = len(state_batches)

        # Store 0th (iteration) estimate as (x0,A0)
        A0,x0  = center(E)
        # Init iterations
        w      = np.tile( zeros(N) , (nBatch,1) )
        Tinv   = np.tile( eye(N)   , (nBatch,1,1) )
        T      = np.tile( eye(N)   , (nBatch,1,1) )

        # Loop iterations
        for iteration in arange(nIter):

            # Assemble current estimate of E[kObs-Lag]
            for ib, ii in enumerate(state_batches):
              E[:,ii] = x0[ii] + w[ib]@A0[:,ii]+ T[ib]@A0[:,ii]

            # Forecast
            for kDAW in DAW:                           # Loop Lag cycles
              for k,t,dt in chrono.cycle(kDAW):        # Loop dkObs steps (1 cycle)
                E = Dyn(E,t-dt,dt)                     # Forecast 1 dt step (1 dkObs)

            if iteration==0:
              stats.assess(k,kObs,'f',E=E)

            # Analysis of y[kObs] (already assim'd [:kObs])
            y    = yy[kObs]
            Y,xo = center(Obs(E,t))
            # Transform obs space
            Y  = Y        @ R.sym_sqrt_inv.T
            dy = (y - xo) @ R.sym_sqrt_inv.T

            # Inflation estimation.
            # Set "effective ensemble size", za = (N-1)/pre-inflation^2.
            if upd_a == 'Sqrt':
                za = N1 # no inflation
            elif upd_a == '-N':
              # Careful not to overwrite w,T,Tinv !
              V,s,UT = svd0(Y)
              grad   = Y@dy
              Pw     = (V * (pad0(s**2,N) + N1)**-1.0) @ V.T
              w_glob = Pw@grad 
              za     = zeta_a(*hyperprior_coeffs(s,N,xN), w_glob)
            else: raise KeyError("upd_a: '" + upd_a + "' not matched.") 

            for ib, ii in enumerate(state_batches):
              # Shift indices (to adjust for time difference)
              ii_kObs = Obs.loc_shift(ii, DAW_dt)
              # Localize
              jj, tapering = obs_taperer(ii_kObs)
              if len(jj) == 0: continue
              Y_jj   = Y[:,jj] * sqrt(tapering)
              dy_jj  = dy[jj]  * sqrt(tapering)

              # "Uncondition" the observation anomalies
              # (and yet this linearization of Obs.mod improves with iterations)
              Y_jj     = Tinv[ib] @ Y_jj
              # Prepare analysis: do SVD
              V,s,UT   = svd0(Y_jj)
              # Gauss-Newton ingredients
              grad     = -Y_jj@dy_jj + w[ib]*za
              Pw       = (V * (pad0(s**2,N) + za)**-1.0) @ V.T
              # Conditioning for anomalies (discrete linearlizations)
              T[ib]    = (V * (pad0(s**2,N) + za)**-0.5) @ V.T * sqrt(N1)
              Tinv[ib] = (V * (pad0(s**2,N) + za)**+0.5) @ V.T / sqrt(N1)
              # Gauss-Newton step
              dw       = Pw@grad
              w[ib]   -= dw

            # Stopping condition # TODO
            #if np.linalg.norm(dw) < N*1e-4:
              #break

        # Analysis 'a' stats for E[kObs].
        stats.assess(k,kObs,'a',E=E)
        stats.trHK [kObs] = trace(Y.T @ Pw @ Y)/HMM.Ny
        stats.infl [kObs] = sqrt(N1/za)
        stats.iters[kObs] = iteration+1

        # Final (smoothed) estimate of E[kObs-Lag]
        for ib, ii in enumerate(state_batches):
          E[:,ii] = x0[ii] + w[ib]@A0[:,ii]+ T[ib]@A0[:,ii]

        E = post_process(E,infl,rot)

        # Forecast smoothed ensemble by shift (1*dkObs)
        if DAW_0 >= 0:
          for k,t,dt in chrono.cycle(DAW_0):
            stats.assess(k-1,None,'u',E=E)
            E = Dyn(E,t-dt,dt)

    # Assess the last (Lag-1) obs ranges
    for kDAW in arange(DAW[0]+1,KObs+1):
      for k,t,dt in chrono.cycle(kDAW):
        stats.assess(k-1,None,'u',E=E)
        E = Dyn(E,t-dt,dt)
    stats.assess(chrono.K,None,'u',E=E)

  return assimilator



@DA_Config
def Var4D(B=None,xB=1.0,Lag=1,wtol=0,nIter=10,**kwargs):
  """4D-Var.

  Cycling scheme is same as in iEnKS (i.e. the shift is always 1*kObs).

  This implementation does NOT do gradient decent (nor quasi-Newton)
  in an inner loop, with simplified models.
  Instead, each (outer) iteration is computed
  non-iteratively as a Gauss-Newton step.
  Thus, since the full (approximate) Hessian is formed,
  there is no benefit to the adjoint trick (back-propagation).
  => This implementation is not suited for big systems.

  Incremental formulation is used, so the formulae look like the ones in iEnKS.

  Settings for reproducing literature benchmarks may be found in
  :mod:`dapper.mods.Lorenz95.sak08`
  """
  def assimilator(stats,HMM,xx,yy):
    Dyn,Obs,chrono,X0,R,KObs = HMM.Dyn, HMM.Obs, HMM.t, HMM.X0, HMM.Obs.noise.C, HMM.t.KObs
    Rm12 = R.sym_sqrt_inv
    Nx = Dyn.M

    # Set background covariance. Note that it is static (compare to iEnKS).
    nonlocal B
    if B is None:
      B = eye(Nx)
    B *= xB
    B12 = CovMat(B).sym_sqrt 

    # Init
    x = X0.mu
    stats.assess(0,mu=x,Cov=B)

    # Loop over DA windows (DAW).
    for kObs in progbar(arange(-1,KObs+Lag+1)):
        kLag = kObs-Lag
        DAW = range( max(0, kLag+1), min(kObs, KObs) + 1)

        # Assimilation (if ∃ "not-fully-assimlated" obs).
        if 0 <= kObs <= KObs:  

            # Init iterations.
            w   = zeros(Nx)          # Control vector for the mean state.
            x0  = x.copy()           # Increment reference.

            for iteration in arange(nIter):
                # Reconstruct smoothed state.
                x = x0 + B12@w
                X = B12 # Aggregate composite TLMs onto B12
                # Forecast.
                for kCycle in DAW:
                  for k,t,dt in chrono.cycle(kCycle):
                    X = Dyn.jacob(x,t-dt,dt) @ X
                    x = Dyn      (x,t-dt,dt)

                # Assess forecast stats
                if iteration==0:
                  stats.assess(k,kObs,'f',mu=x,Cov=X@X.T)

                # Observe.
                Y  = Obs.jacob(x,t) @ X
                xo = Obs(x,t)

                # Analysis prep.
                y      = yy[kObs]          # Get current obs.
                dy     = Rm12 @ (y - xo)   # Transform obs space.
                Y      = Rm12 @ Y          # Transform obs space.
                V,s,UT = svd0(Y.T)         # Decomp for lin-alg update comps.

                # Post. cov (approx) of w, estimated at current iteration, raised to power.
                Cow1 = (V * (pad0(s**2,Nx) + 1)**-1.0) @ V.T

                # Compute analysis update.
                grad = Y.T@dy - w          # Cost function gradient
                dw   = Cow1@grad           # Gauss-Newton step
                w   += dw                  # Step

                if dw@dw < wtol*Nx:
                  break
            # END loop iteration

            # Assess (analysis) stats.
            final_increment = X@dw
            stats.assess(k,   kObs, 'a', mu=x+final_increment, Cov=X@Cow1@X.T)
            stats.iters      [kObs] = iteration+1

            # Final (smoothed) estimate at [kLag].
            x = x0 + B12@w
            X = B12
        # END assimilation block


        # Slide/shift DAW by propagating smoothed ('s') state from [kLag].
        if -1 <= kLag < KObs: 
          if kLag>=0: stats.assess(chrono.kkObs[kLag],kLag,'s',mu=x,Cov=X@Cow1@X.T)
          for k,t,dt in chrono.cycle(kLag+1):
            stats.assess(k-1,None,'u',mu=x,Cov=Y@Y.T)
            X = Dyn.jacob(x,t-dt,dt) @ X
            x = Dyn      (x,t-dt,dt)

    # END loop kObs
    stats.assess(k,KObs,'us',mu=x,Cov=X@Cow1@X.T)
  return assimilator




