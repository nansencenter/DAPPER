# DA filters with different inflation estimation methods.
# This file is quite messy (as compared to DAPPER/da_methods.py, for example),
# having been used for trying out many different methods.

from common import *

@DA_Config
def EnKF_pre(upd_a,N,infl=1.0,rot=False,**kwargs):
  """
  As EnKF(), except with inflation pre-analysis instead of post-.
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
        E = post_process(E,infl,rot)
        E = EnKF_analysis(E,h(E,t),h.noise,yy[kObs],upd_a,stats,kObs)

      stats.assess(k,kObs,E=E)
  return assimilator


@DA_Config
def EAKF_A07(N,var_f=None,damp=0.9,CLIP=0.9,ordr='rand',
    infl=1.0,rot=False,**kwargs):
  """
  Based on 
    - (My reading of) Anderson 2007: "An adaptive covariance inflation
      error correction algorithm for ensemble filters"
    - da_methods.py:EAKF().
  Bug test: set var_f really small (1e-6) to disable adaptive inflation,
            and check that it reproduces scores of the EnKF('Sqrt')
            with fixed/tuned infl.
  """

  def assimilator(stats,twin,xx,yy):
    f,h,chrono,X0 = twin.f, twin.h, twin.t, twin.X0
    N1   = N-1
    R    = h.noise
    Rm12 = h.noise.C.sym_sqrt_inv

    nonlocal var_f
    if var_f:
      estimate_var_f = False
    else: 
      estimate_var_f = True
      var_f = 10 # Var  of prior inflation^2. Init.
    b2 = 1.0     # Mean of prior inflation^2. Init.

    E = X0.sample(N)
    stats.assess(0,E=E)

    stats.var_f = zeros(chrono.KObs+1)

    for k,kObs,t,dt in progbar(chrono.forecast_range):
      E = f(E,t-dt,dt)
      E = add_noise(E, dt, f.noise, kwargs)

      if kObs is not None:
        stats.assess(k,kObs,'f',E=E)
        y    = yy[kObs]
        inds = serial_inds(ordr, y, R, anom(E)[0])
            
        for UPDATE_TARGET in ['INFL','STATE']:
          # one pass for estimating inflation, and then
          # one pass for the actual state update
          if UPDATE_TARGET is 'STATE':
            # Dampen to background
            _b2 = damp*b2 + (1-damp)*1.0
            _b2 = max(CLIP,_b2)
            # Inflate
            E = post_process(E,infl*sqrt(_b2),0)
            # 
            stats.infl [kObs] = sqrt(b2)
            stats.var_f[kObs] = var_f

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
            su    = 1/( 1/skk + 1/N1 )
            alpha = (N1/(N1+skk))**(0.5)
            #
            dy2   = su*dyj/N1 # (mean is absorbed in dyj)
            Y2    = alpha*Yj

            if UPDATE_TARGET=='INFL':
              # See Anderson'2007, in particular eqn A.8.
              OLD = b2                           # forecast (_f) mean of b2
              D   = dyj                          # observed mean innovation
              so  = 1                            # "sigma_o^2" = 1 coz of Rm12 above
              sp  = skk/N1                       # "sigma_p^2"
              T2  = lambda b2: b2*sp + so        # theta^2(b2)
              zz  = np.roots([1, -T2(OLD), .5*var_f*sp**2, -.5*var_f*(sp*D)**2])
              zz  = zz[np.isreal(zz)].real       # restrict to reals
              bb2 = (zz-so)/sp                   # Solve zz = T2(bb2)
              b2  = bb2[abs(bb2-OLD).argmin()]   # Posterior mean b2: pick closest to OLD

              # Fit posterior var via ratio of posterior eval'd at pt1/pt2
              # See Anderson'2007 eqn 5.10.
              if estimate_var_f: 
                pt1   = b2+sqrt(var_f)
                pt2   = b2
                Rtio  = exp(( (pt2-OLD)**2 - (pt1-OLD)**2 )/(2*var_f)) # prior
                Rtio *= sqrt(T2(pt2)/T2(pt1))                 # lklhd determinant
                Rtio *= exp( D**2/2*(1/T2(pt2) - 1/T2(pt1)) ) # lklhd exponent
                Rtio  = min(0.9999,Rtio)                      # clip
                var_f = -var_f / 2 / log(Rtio)                # Eqn 5.10

            elif UPDATE_TARGET=='STATE':
              if skk<1e-9: continue
              # Update state (regress from y2)
              Regression = A.T @ Yj/np.sum(Yj**2)
              mu        += Regression*dy2
              A         += np.outer(Y2 - Yj, Regression)
              E          = mu + A

        E = post_process(E,1.0,rot)
      stats.assess(k,kObs,E=E)
  return assimilator


@DA_Config
def ETKF_M11(N,var_f,var_o=None,CLIP=0.9,damp=1.0,
    infl=1.0,rot=False,**kwargs):
  """
  Based on 
    - Miyoshi 2011: "The Gaussian Approach to
      Adaptive Covariance Inflation and Its Implementation
      with the Local Ensemble Transform Kalman Filter"
    - Wang 2003: "A Comparison of Breeding and Ensemble Transform
      Kalman Filter Ensemble Forecast Schemes"
    - Sqrt-EnKF (i.e. ETKF) code from da_methods.py.
  Bug test: set var_f really small (1e-6) to disable adaptive inflation,
            and check that it reproduces scores of the EnKF('Sqrt')
            with fixed/tuned infl.

  M11 also uses 1.5% fixed inflation.

  Finally, after reading Hunt2007"efficient" which Miyoshi makes reference to,
  it seems that they do also apply inflation when computing the mean,
  and not just the anomalies. Thus this here implementation is OBSOLETE.
  """
    
  def assimilator(stats,twin,xx,yy):
    f,h,chrono,X0,R  = twin.f, twin.h, twin.t, twin.X0, twin.h.noise.C

    # Init
    E = X0.sample(N)
    stats.assess(0,E=E)

    nonlocal var_o
    estimate_v_o = True if not var_o else False
    stats.var_o = zeros(chrono.KObs+1)
    b2 = 1.0

    # Loop
    for k,kObs,t,dt in progbar(chrono.forecast_range):
      E = f(E,t-dt,dt)
      E = add_noise(E, dt, f.noise, kwargs)

      # Analysis update
      if kObs is not None:
        stats.assess(k,kObs,'f',E=E)

        mu = mean(E,0)
        A  = E - mu

        hE = h(E,t)
        y  = yy[kObs]
        hx = mean(hE,0)
        Y  = hE-hx
        dy = y - hx

        V,s,_ = svd0(Y @ R.sym_sqrt_inv.T)
        d     = pad0(s**2,N) + (N-1)
        Pw    = ( V * d**(-1.0) ) @ V.T
        T     = ( V * d**(-0.5) ) @ V.T * sqrt(N-1) 

        # Infl estimation
        ###############
        dR = dy @ R.sym_sqrt_inv.T
        YR = Y  @ R.sym_sqrt_inv.T

        trHPHR = trace(YR.T @ YR)/(N-1)
        b2_o   = (dR@dR - h.m)/trHPHR

        if estimate_v_o:
          var_o = 2/h.m * ( (b2*trHPHR + h.m)/trHPHR )**2
          stats.var_o[kObs] = var_o

        b2 = (b2/var_f + b2_o/var_o)/(1/var_f + 1/var_o)
        stats.infl[kObs] = sqrt(b2)

        # Ad-hoc stuff
        _b2 = damp*b2 + (1-damp)*1.0
        _b2 = max(CLIP,_b2)

        w = dy @ R.inv @ Y.T @ Pw # infl not used here
        E = mu + w@A + (infl*sqrt(_b2)*T)@A

        E = post_process(E,1.0,rot)
      stats.assess(k,kObs,E=E)
  return assimilator



@DA_Config
def EnKF_N_mod(N,L=np.inf,nu_f=None,nu_o=1,nu0=100,Cond=True,
    deb=False,CLIP=0.9,damp=1.0,
    xN=1.0,g=0,infl=1.0,rot=False,**kwargs):
  """
  """
    
  def assimilator(stats,twin,xx,yy):
    f,h,chrono,X0,R  = twin.f, twin.h, twin.t, twin.X0, twin.h.noise.C

    if deb: debias = lambda b2: b2*(N*h.m - 2)/(N*h.m) - 1/N
    else:   debias = lambda b2: b2

    N1         = N-1      # Abbrev
    eN_        = (N+1)/N  # Effect of unknown mean
    cL_        = (N+g)/N1 # Coeff in front of log term
    prior_mode = eN_/cL_  # Mode of l1 (un-corrected)

    # Init
    E = X0.sample(N)
    stats.assess(0,E=E)

    b2 = 1.0
    nonlocal nu_f, L
    if nu_f is None: nu_f = nu0           # init
    if L    is None: L    = equiv_L(nu_f) # yields fixed nu_f

    stats.nu_f = np.full(chrono.KObs+1,nan)

    # Loop
    for k,kObs,t,dt in progbar(chrono.forecast_range):
      E = f(E,t-dt,dt)
      E = add_noise(E, dt, f.noise, kwargs)

      # Analysis update
      if kObs is not None:
        stats.assess(k,kObs,'f',E=E)

        mu = mean(E,0)
        A  = E - mu

        hE = h(E,t)
        y  = yy[kObs]
        hx = mean(hE,0)
        Y  = hE-hx
        dy = y - hx

        dR = dy @ R.sym_sqrt_inv.T
        YR = Y  @ R.sym_sqrt_inv.T

        V,s,UT = svd0(YR)
        du     = UT @ dR 
        dgn_N  = lambda l1: pad0( (l1*s)**2, N ) + N1

        # EnKF-N (a) given prior b
        ##################
        mc  = mode_adjust(prior_mode,dgn_N,N1)
        xN_ = xN
        eN  = eN_*xN_/mc
        cL  = cL_*xN_*mc

        # Make dual cost function (in terms of l1)
        pad_rk = lambda arr: pad0( arr, min(N,h.m) )
        dgn_rk = lambda l1: pad_rk((l1*s)**2) + N1

        J      = lambda b1: np.sum(du**2/dgn_rk(b1)) \
                 +    eN        / (eN+nu_f/N1) /b1**2 \
                 +    cL        / (eN+nu_f/N1) *log(b1**2) \
                 +    nu_f/N1*b2/ (eN+nu_f/N1) /b1**2 \
                 +    nu_f/N1   / (eN+nu_f/N1) *log(b1**2)
        # Find inflation factor (optimize)
        b2 = fmin_bfgs(J, x0=[1], gtol=1e-4, disp=0)**2
        nu_f = (nu_f + nu_o) * exp(-1/L)

        stats.infl[kObs] = sqrt(b2)
        stats.nu_f[kObs] = nu_f

        # Ad-hoc stuff
        _b2 = damp*b2 + (1-damp)*1.0
        _b2 = max(CLIP,_b2)

        # Set inflation used in this update
        za = N1/infl**2/_b2

        # ETKF
        ##################
        Pw   = (V * (pad0(s**2,N) + za)**-1.0) @ V.T
        T    = (V * (pad0(s**2,N) + za)**-0.5) @ V.T * sqrt(N1)
        w    = dy @ R.inv @ Y.T @ Pw
        E    = mu + w@A + T@A

        E = post_process(E,1.0,rot)
      stats.assess(k,kObs,E=E)
  return assimilator



@DA_Config
def ETKF_Xplct(N,L=np.inf,nu_f=None,nu_o1=True,nu0=100,deb=False,damp=1.0,
    CLIP=0.9,infl=1.0,rot=False,**kwargs):
  """
  """
    
  def assimilator(stats,twin,xx,yy):
    f,h,chrono,X0,R  = twin.f, twin.h, twin.t, twin.X0, twin.h.noise.C
    N1 = N-1
    if deb: debias = lambda b2: b2*(N*h.m - 2)/(N*h.m) - 1/N
    else:   debias = lambda b2: b2

    # Init
    E = X0.sample(N)
    stats.assess(0,E=E)

    b2 = 1.0
    nonlocal nu_f, L
    if nu_f is None: nu_f = nu0           # init
    if L    is None: L    = equiv_L(nu_f) # yields fixed nu_f
    if nu_o1       : nu_o = 1.0

    stats.nu_f = np.full(chrono.KObs+1,nan)
    stats.nu_o = np.full(chrono.KObs+1,nan)

    # Loop
    for k,kObs,t,dt in progbar(chrono.forecast_range):
      E = f(E,t-dt,dt)
      E = add_noise(E, dt, f.noise, kwargs)

      # Analysis update
      if kObs is not None:
        stats.assess(k,kObs,'f',E=E)

        mu = mean(E,0)
        A  = E - mu

        hE = h(E,t)
        y  = yy[kObs]
        hx = mean(hE,0)
        Y  = hE-hx
        dy = y - hx

        dR = dy @ R.sym_sqrt_inv.T
        YR = Y  @ R.sym_sqrt_inv.T
        V,s,UT = svd0(YR)

        # Infl estimation
        ##################
        trHPHR = trace(YR.T @ YR)/N1 # sum(s**2)/N1
        b2_o   = (dR@dR - h.m)/trHPHR
        b2_o   = debias(b2_o)

        # Estimate obs certainty
        if not nu_o1:
          PsiBe = b2*trHPHR/h.m
          nu_o  = h.m * ( PsiBe/(PsiBe + 1) )**2
          stats.nu_o[kObs] = nu_o

        # Bayes rule for inflation
        b2     = (b2*nu_f + b2_o*nu_o)/(nu_f + nu_o)
        nu_f   = (nu_f + nu_o) * exp(-1/L) # forecast/analysis in one go

        stats.infl[kObs] = sqrt(b2)
        stats.nu_f[kObs] = nu_f

        # Ad-hoc stuff
        _b2 = damp*b2 + (1-damp)*1.0
        _b2 = max(CLIP,_b2)

        # ETKF
        ##################
        za   = N1/infl**2/_b2
        Pw   = (V * (pad0(s**2,N) + za)**-1.0) @ V.T
        T    = (V * (pad0(s**2,N) + za)**-0.5) @ V.T * sqrt(N1)
        w    = dy @ R.inv @ Y.T @ Pw
        E    = mu + w@A + T@A

        E = post_process(E,1.0,rot)
      stats.assess(k,kObs,E=E)
  return assimilator


@DA_Config
def ETKF_InvCS(N,Uni=True,Var=False,pt='mean',L=np.inf,nu0=1000,deb=False,damp=1.0,
    CLIP=0.9,infl=1.0,rot=False,**kwargs):
  """
  """
    
  def assimilator(stats,twin,xx,yy):
    f,h,chrono,X0,R  = twin.f, twin.h, twin.t, twin.X0, twin.h.noise.C
    N1 = N-1
    if deb: debias = lambda b2: 1 + 1/N/b2
    else:   debias = lambda b2: 1

    # Init
    E = X0.sample(N)
    stats.assess(0,E=E)
    iC2 = InvChi2Filter(sc=1.0,nu=nu0,L=L)

    # should be nu_a, but does not matter much
    stats.nu_f = np.full(chrono.KObs+1,nan)

    # Loop
    for k,kObs,t,dt in progbar(chrono.forecast_range):
      E = f(E,t-dt,dt)
      E = add_noise(E, dt, f.noise, kwargs)

      iC2.forecast()

      # Analysis update
      if kObs is not None:
        stats.assess(k,kObs,'f',E=E)

        mu = mean(E,0)
        A  = E - mu

        hE = h(E,t)
        y  = yy[kObs]
        hx = mean(hE,0)
        Y  = hE-hx
        dy = y - hx

        dR = dy @ R.sym_sqrt_inv.T
        YR = Y  @ R.sym_sqrt_inv.T
        V,s,UT = svd0(YR)
        du     = UT @ dR 

        # Infl estimation
        ##################
        # Define likelihood
        if Uni:
          trHPHR    = debias(iC2.mean)*trace(YR.T @ YR)/N1 # sum(s**2)/N1
          log_lklhd = lambda b2: Chi2_logp(h.m + trHPHR*b2, h.m, dR@dR)
        else:
          dgn_v     = diag_HBH_I(sqrt(debias(iC2.mean))*s/sqrt(N1),min(N,h.m))
          log_lklhd = lambda b2: diag_Gauss_logp(0, dgn_v(b2), du).sum(axis=1)
        # BayesRule
        if Var: iC2.log_Var_update(log_lklhd)
        else:   iC2.log_update    (log_lklhd)

        b2 = getattr(iC2,pt) # 'mean','mode','sc'
        stats.infl[kObs] = sqrt(b2)
        stats.nu_f[kObs] = iC2.nu

        # Ad-hoc stuff
        _b2 = damp*b2 + (1-damp)*1.0
        _b2 = max(CLIP,_b2)

        # ETKF
        ##################
        za   = N1/infl**2/_b2
        Pw   = (V * (pad0(s**2,N) + za)**-1.0) @ V.T
        T    = (V * (pad0(s**2,N) + za)**-0.5) @ V.T * sqrt(N1)
        w    = dy @ R.inv @ Y.T @ Pw
        E    = mu + w@A + T@A

        E = post_process(E,1.0,rot)
      stats.assess(k,kObs,E=E)
  return assimilator



# Obsolete: replaced by hyperprior_coeffs
def mode_adjust(prior_mode,dgn_N,N1):
  # As a func of I-KH ("prior's weight"), adjust l1's mode towards 1.
  # Note: I-HK = mean( dgn_N(1.0)**(-1) )/N ≈ 1/(1 + HBH/R).
  I_KH  = mean( dgn_N(1.0)**(-1) )*N1 # Normalize by f.m ?
  #I_KH = 1/(1 + (s**2).sum()/N1)     # Alternative: use tr(HBH/R).
  mc    = sqrt(prior_mode**I_KH)      # "mode correction".
  return mc


from scipy.optimize import fmin_bfgs

@DA_Config
def EnKF_N_InvCS(N,g2=0,joint=False,pt='mean',
    Cond=True,Uni=True,VO=False,L=np.inf,nu0=1000,CLIP=0.9,damp=1.0,
    xN=1.0,g=0,infl=1.0,rot=False,**kwargs):
  """
  """
    
  def assimilator(stats,twin,xx,yy):
    f,h,chrono,X0,R  = twin.f, twin.h, twin.t, twin.X0, twin.h.noise.C

    N1         = N-1      # Abbrev
    eN_        = (N+1)/N  # Effect of unknown mean
    cL_        = (N+g)/N1 # Coeff in front of log term
    prior_mode = eN_/cL_  # Mode of l1 (un-corrected)

    # Init
    E = X0.sample(N)
    stats.assess(0,E=E)
    iC2 = InvChi2Filter(sc=1.0,nu=nu0,L=L)

    stats.a    = np.full(chrono.KObs+1,nan)
    # should be nu_a, but does not matter much
    stats.nu_f = np.full(chrono.KObs+1,nan)

    # Loop
    for k,kObs,t,dt in progbar(chrono.forecast_range):
      E = f(E,t-dt,dt)
      E = add_noise(E, dt, f.noise, kwargs)

      iC2.forecast()

      # Analysis update
      if kObs is not None:
        stats.assess(k,kObs,'f',E=E)

        mu = mean(E,0)
        A  = E - mu

        hE = h(E,t)
        y  = yy[kObs]
        hx = mean(hE,0)
        Y  = hE-hx
        dy = y - hx

        dR = dy @ R.sym_sqrt_inv.T
        YR = Y  @ R.sym_sqrt_inv.T

        V,s,UT = svd0(YR)
        du     = UT @ dR 
        dgn_N  = lambda l1: pad0( (l1*s)**2, N ) + N1


        #############
        mc  = mode_adjust(prior_mode,dgn_N,N1)
        xN_ = xN
        eN  = eN_*xN_/mc
        cL  = cL_*xN_*mc

        # Make dual cost function (in terms of l1)
        pad_rk = lambda arr: pad0( arr, min(N,h.m) )
        dgn_rk = lambda l1: pad_rk((l1*s)**2) + N1

        if joint:
          # EnKF-N in (a,b)
          J      = lambda ab: np.sum(du**2/dgn_rk(ab[0]*ab[1])) \
                   +    eN/ab[0]**2 \
                   +    cL*log(ab[0]**2) \
                   +    iC2.nu*iC2.sc /N1/ab[1]**2 \
                   +    (iC2.nu-g2)   /N1*log(ab[1]**2)
          a,b = fmin_bfgs(J, x0=[1,sqrt(iC2.mean)], gtol=1e-4, disp=0)
        else:
          # EnKF-N in (a|b)
          J      = lambda a: np.sum(du**2/dgn_rk(a*sqrt(iC2.mean))) \
                   +    eN/a**2 \
                   +    cL*log(a**2)
          a = fmin_bfgs(J, x0=[1], gtol=1e-4, disp=0)

        # Estimating b given a
        ##################
        _a = a if Cond else 1.0
        if Uni:
          trHPHR    = _a**2*trace(YR.T @ YR)/N1 # sum(s**2)/N1
          log_lklhd = lambda b2: Chi2_logp(h.m + trHPHR*b2, h.m, dR@dR)
        else:
          dgn_v     = diag_HBH_I(_a*s/sqrt(N1),min(N,h.m))
          log_lklhd = lambda b2: diag_Gauss_logp(0, dgn_v(b2), du).sum(axis=1)
        iC2.log_update(log_lklhd)

        # Option: "var only": use b from -N estim,
        # i.e. only use variance from iC2 update.
        # Seems beneficial if Uni
        if VO: iC2.sc = b**2

        # Choose b
        b2 = getattr(iC2,pt) # 'mean','mode','sc'

        stats.a   [kObs] = a
        stats.infl[kObs] = a*sqrt(b2)
        stats.nu_f[kObs] = iC2.nu

        ab2  = a**2 * b2
        # Ad-hoc stuff
        _ab2 = damp*ab2 + (1-damp)*1.0
        _ab2 = max(CLIP,_ab2)

        # ETKF
        ##################
        za   = N1/infl**2/_ab2
        Pw   = (V * (pad0(s**2,N) + za)**-1.0) @ V.T
        T    = (V * (pad0(s**2,N) + za)**-0.5) @ V.T * sqrt(N1)
        w    = dy @ R.inv @ Y.T @ Pw
        E    = mu + w@A + T@A

        E = post_process(E,1.0,rot)
      stats.assess(k,kObs,E=E)
  return assimilator



@DA_Config
def EnKF_N_Xplct(N,L=np.inf,nu_f=None,nu_o1=True,nu0=100,Cond=True,
    deb=False,CLIP=0.9,damp=1.0,
    xN=1.0,g=0,infl=1.0,rot=False,**kwargs):
  """
  """
    
  def assimilator(stats,twin,xx,yy):
    f,h,chrono,X0,R  = twin.f, twin.h, twin.t, twin.X0, twin.h.noise.C

    if deb: debias = lambda b2: b2*(N*h.m - 2)/(N*h.m) - 1/N
    else:   debias = lambda b2: b2

    N1         = N-1      # Abbrev
    eN_        = (N+1)/N  # Effect of unknown mean
    cL_        = (N+g)/N1 # Coeff in front of log term
    prior_mode = eN_/cL_  # Mode of l1 (un-corrected)

    # Init
    E = X0.sample(N)
    stats.assess(0,E=E)

    # NB Cannot safely leave nu_f arg unspecified coz it's persistent
    #    through runs coz of nonlocal keyword.
    b2 = 1.0
    nonlocal nu_f, L
    if nu_f is None : nu_f = nu0           # init
    if L    is None : L    = equiv_L(nu_f) # yields fixed nu_f
    if nu_o1        : nu_o = 1.0

    stats.a    = np.full(chrono.KObs+1,nan)
    stats.b    = np.full(chrono.KObs+1,nan)
    stats.nu_f = np.full(chrono.KObs+1,nan)
    stats.nu_o = np.full(chrono.KObs+1,nan)

    # Loop
    for k,kObs,t,dt in progbar(chrono.forecast_range):
      E = f(E,t-dt,dt)
      E = add_noise(E, dt, f.noise, kwargs)

      # Analysis update
      if kObs is not None:
        stats.assess(k,kObs,'f',E=E)

        mu = mean(E,0)
        A  = E - mu

        hE = h(E,t)
        y  = yy[kObs]
        hx = mean(hE,0)
        Y  = hE-hx
        dy = y - hx

        dR = dy @ R.sym_sqrt_inv.T
        YR = Y  @ R.sym_sqrt_inv.T

        V,s,UT = svd0(YR)
        du     = UT @ dR 
        dgn_N  = lambda l1: pad0( (l1*s)**2, N ) + N1

        # EnKF-N (a) given prior b
        ##################
        mc  = mode_adjust(prior_mode,dgn_N,N1)
        xN_ = xN
        eN  = eN_*xN_/mc
        cL  = cL_*xN_*mc

        # Make dual cost function (in terms of l1)
        pad_rk = lambda arr: pad0( arr, min(N,h.m) )
        dgn_rk = lambda l1: pad_rk((l1*s)**2) + N1

        J      = lambda a: np.sum(du**2/dgn_rk(a*sqrt(b2))) \
                 +    eN/a**2 \
                 +    cL*log(a**2)
        # Find inflation factor (optimize)
        a = fmin_bfgs(J, x0=[1], gtol=1e-4, disp=0)
        # TODO: speed-up optimization with analytical derivatives

        # Estimating b given a
        ##################
        _a = a if Cond else 1.0
        trHPHR = _a**2*trace(YR.T @ YR)/N1 # sum(s**2)/N1
        b2_o   = (dR@dR - h.m)/trHPHR
        b2_o   = debias(b2_o)

        # Estimate obs certainty
        if not nu_o1:
          PsiBe = b2*trHPHR/h.m
          nu_o  = h.m * ( PsiBe/(PsiBe + 1) )**2
          stats.nu_o[kObs] = nu_o

        # Bayes rule for inflation
        b2     = (b2*nu_f + b2_o*nu_o)/(nu_f + nu_o)
        nu_f   = (nu_f + nu_o) * exp(-1/L)

        stats.a   [kObs] = a
        stats.b   [kObs] = sqrt(b2)
        stats.infl[kObs] = a*sqrt(b2)
        stats.nu_f[kObs] = nu_f

        ab2  = a**2 * b2
        # Ad-hoc stuff
        _ab2 = damp*ab2 + (1-damp)*1.0
        _ab2 = max(CLIP,_ab2)

        # Set inflation used in this update
        za = N1/infl**2/_ab2

        # ETKF
        ##################
        Pw   = (V * (pad0(s**2,N) + za)**-1.0) @ V.T
        T    = (V * (pad0(s**2,N) + za)**-0.5) @ V.T * sqrt(N1)
        w    = dy @ R.inv @ Y.T @ Pw
        E    = mu + w@A + T@A

        E = post_process(E,1.0,rot)
      stats.assess(k,kObs,E=E)
  return assimilator








def diag_HBH_I(spectrum_HBH, rk):
  """
  Returns a function (of l2) that produces
  a matrix where each row [i] (of length rk) is
  the vector l2[i]*(sigma)^2 + 1.
  """
  def _pad(ss,N):
      out = zeros((ss.shape[0],N))
      out[:,:ss.shape[1]] = ss
      return out
  def fun(l2):
    l2 = np.atleast_1d(l2) # support singleton input 
    # l2 gets broadcasted for spectrum_HBH along axis 1
    diags = _pad(l2[:,None]*spectrum_HBH**2, rk) + 1
    return diags
  return fun






def Chi2_logp(sc,nu,x):
  # Correct up to a constant (that does not depend on sc or x)
  return ( nu/2-1)*log(x/sc) -(nu*x/2)/sc - log(sc)

def iChi2_logp(sc,nu,x):
  # Correct up to a constant (that does not depend on sc or x)
  return (-nu/2-1)*log(x/sc) -(nu*sc/2)/x - log(sc)

def diag_Gauss_logp(mu, sigma2, x):
  # Correct up to a constant (that does not depend on mu,sigma2, or x)
  if np.any(sigma2<0):
    print(sigma2)
  J = log(sigma2) + (x-mu)**2/sigma2
  return -0.5*J


from scipy.optimize import brenth
from scipy.optimize import minimize_scalar as minz
#from scipy.integrate import quad
class InvChi2Filter(MLR_Print):
  def __init__(self,sc=1.0,nu=5,L=np.inf):
    """
    - nu: certainty; start at nu=5 such that e.g. variance exists.
    - sc: scale param.
    - L : forgetting time scale
    """
    self.nu     = nu
    self.sc     = sc
    self.forget = exp(-1/L)

  @property
  def mean(self): return self.sc * self.nu/(self.nu - 2)
  @property
  def mode(self): return self.sc * self.nu/(self.nu + 2)
  @property
  def var (self): return 2*self.mean**2/(self.nu - 4)

  def domain(self,R=1e3):
    # Find bounds such that pdf(mode)/pdf(bound) = R
    # For maximum efficiency, this could be had from table lookup (func of nu).
    f     = lambda x: self.log_pdf(self.mode) - self.log_pdf(x) - log(R)
    Lower = brenth(f, 1e-8     , self.mode, xtol=1e-6)
    Upper = brenth(f, self.mode, 1000,      xtol=1e-6)
    return Lower, Upper
    #return 0.1, 10 # This is usually a safe shortcut

  def log_pdf(self,x):
    return iChi2_logp(self.sc,self.nu, x)
  
  def forecast(self,k=1):
    self.nu *= self.forget**k
    #self.sc = 1.0 + self.forget**k*(self.sc - 1.0)

  def log_update(self,log_lklhd):
    for RUN in ['VALIDATION','ACTUAL']:
      if RUN is 'VALIDATION':
        # Make posterior==prior
        LL = lambda x: np.ones_like(x)
      else:
        LL = log_lklhd

      log_post  = lambda x: self.log_pdf(x) + LL(x)
      post_nn   = lambda x: exp(log_post(x) - log_post(self.sc))
      # Version with a single pdf evaluation
      xx        = linspace(*self.domain(), 1001)
      pp        = post_nn(xx)
      normlzt   = sum(pp)
      mean      = sum(          xx*pp)/normlzt
      var       = sum((xx-mean)**2*pp)/normlzt
      # Version using quadrature functions
      #_quad     = lambda f: sum(f(xx))*(xx[1]-xx[0]) # manual
      #_quad     = lambda f: quad(f,*self.domain())[0]  # built-in
      #normlzt   = _quad(lambda x: post_nn(x))
      #post      = lambda x: post_nn(x) / normlzt
      #mean      = _quad(lambda x:  post(x)*x)
      #var       = _quad(lambda x:  post(x)*(x-mean)**2)
      if RUN is 'VALIDATION':
        # Estimate quadrature errors
        err_mean = mean - self.mean
        err_var  = var  - self.var
      else:
        # Update params
        mean      = mean - err_mean
        var       = var  - err_var
        self.sc, self.nu = Gauss_to_iChi2(mean,var)
        # NB post_nn is is affected by the updates => don't use hereafter.


  def log_Var_update(self,log_lklhd):
    # Fit iChi2 mode and J(mode*d)-J(mode)
    log_post  = lambda x: self.log_pdf(x) + log_lklhd(x)
    J         = lambda x: -2*log_post(x)
    #mode      = fmin_bfgs(J, self.mode, gtol=1e-4, disp=0)
    mode      = minz(J, bounds=(0.1,3*self.mode), method='bounded').x
    d         = (1 + sqrt(self.var)/100)/mode
    dJ        = J(mode*d) - J(mode)
    nu        = dJ/(log(d) + 1/d - 1) - 2
    self.nu   = nu
    self.sc   = (nu+2)/nu * mode


def Gauss_to_iChi2(mean, var):
  nu = 4 + 2*mean**2/var
  sc = (nu-2)/nu * mean
  return sc, nu

def iChi2_to_Gauss(sc,nu):
  """Ambiguous (mean!=mode). Don't use."""
  iC2 = InvChi2Filter(sc,nu)
  return iC2.mean, iC2.mode, iC2.var

# Find L such that NU does not change if its forecast/analysis (f/a) cycle is:
# - a: decrease by exp(-1/L)
# - f: increment by 1
# => (1+NU_infy) * exp(-1/L) = NU_infty
equiv_L = lambda NU: 1/log(1 + 1/NU)
# Note: For large NU, equiv_L ≈ NU

# Var of chi1 var in terms of chi2 params.
from scipy.stats import chi
var_Chi1 = lambda nu: chi(df=nu,scale=sqrt(1.0)/sqrt(nu)).var()
# A good approximation is var(chi1) = sc/nu/2 (parameters for chi2).
# Compare with np.var( sqrt ( np.sum( sqrt(sc)*randn((10**4,nu))** 2,axis=1)/nu ) )
var_iChi1 = lambda nu: np.var( sqrt ( 1/(np.sum( randn((10**6//nu,nu))** 2,axis=1)/nu) ) )
# For large nu, can just use the chi1 formula.

