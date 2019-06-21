"""Methods often used to compare against to indicate baselines performance.

Many are based on [Raa16a]_.
"""

from dapper import *

@DA_Config
def EnCheat(**kwargs):
  """A baseline/reference method.
  Should be implemented as part of Stats instead."""
  def assimilator(stats,HMM,xx,yy): pass
  return assimilator


@DA_Config
def Climatology(**kwargs):
  """
  A baseline/reference method.
  Note that the "climatology" is computed from truth, which might be
  (unfairly) advantageous if the simulation is too short (vs mixing time).
  """
  def assimilator(stats,HMM,xx,yy):
    Dyn,Obs,chrono,X0 = HMM.Dyn, HMM.Obs, HMM.t, HMM.X0

    muC = mean(xx,0)
    AC  = xx - muC
    PC  = CovMat(AC,'A')

    stats.assess(0,mu=muC,Cov=PC)
    stats.trHK[:] = 0

    for k,kObs,_,_ in progbar(chrono.ticker):
      fau = 'u' if kObs is None else 'fau'
      stats.assess(k,kObs,fau,mu=muC,Cov=PC)
  return assimilator


@DA_Config
def OptInterp(**kwargs):
  """
  Optimal Interpolation -- a baseline/reference method.
  Uses the Kalman filter equations,
  but with a prior from the Climatology.
  """
  def assimilator(stats,HMM,xx,yy):
    Dyn,Obs,chrono,X0 = HMM.Dyn, HMM.Obs, HMM.t, HMM.X0

    # Get H.
    msg  = "For speed, only time-independent H is supported."
    H    = Obs.jacob(np.nan, np.nan)
    if not np.all(np.isfinite(H)): raise AssimFailedError(msg)

    # Compute "climatological" Kalman gain
    muC = mean(xx,0)
    AC  = xx - muC
    PC  = (AC.T @ AC) / (xx.shape[0] - 1)
    KG  = mrdiv(PC@H.T, H@PC@H.T + Obs.noise.C.full)

    # Setup scalar "time-series" covariance dynamics.
    # ONLY USED FOR DIAGNOSTICS, not to change the Kalman gain.
    Pa    = (eye(Dyn.M) - KG@H) @ PC
    CorrL = estimate_corr_length(AC.ravel(order='F'))
    WaveC = wave_crest(trace(Pa)/trace(2*PC),CorrL)

    # Init
    mu = muC
    stats.assess(0,mu=mu,Cov=PC)

    for k,kObs,t,dt in progbar(chrono.ticker):
      # Forecast
      mu = Dyn(mu,t-dt,dt)
      if kObs is not None:
        stats.assess(k,kObs,'f',mu=muC,Cov=PC)
        # Analysis
        mu = muC + KG@(yy[kObs] - Obs(muC,t))
      stats.assess(k,kObs,mu=mu,Cov=2*PC*WaveC(k,kObs))
  return assimilator


@DA_Config
def Var3D(infl=1.0,**kwargs):
  """
  3D-Var -- a baseline/reference method.

  This implementation is not "Var"-ish, as there is no optimization.
  Instead, it takes one step of the Kalman filter equations,
  where the background covariance is estimated from the Climatology
  (for simplicity, the truth trajectory, xx).

  However, the background does get scaled,
  distinguishing (improving) this method from OptInterp().
  The scaling is estimated by time-series approximations to the dynamics.
  Alternative approach: see Var3D_Lag (archived), which uses the
  auto-cov (at Lag=dtObs) to approx. the conditional covariance.
  """
  def assimilator(stats,HMM,xx,yy):
    Dyn,Obs,chrono,X0 = HMM.Dyn, HMM.Obs, HMM.t, HMM.X0

    # Compute "climatology"
    muC = mean(xx,0)
    AC  = xx - muC
    PC  = (AC.T @ AC)/(xx.shape[0] - 1)

    # TODO: The wave-crest yields good results for sak08, but not for boc10 

    # Setup scalar "time-series" covariance dynamics
    CorrL = estimate_corr_length(AC.ravel(order='F'))
    WaveC = wave_crest(0.5,CorrL) # Nevermind careless W0 init

    # Init
    mu = muC
    P  = PC
    stats.assess(0,mu=mu,Cov=P)

    for k,kObs,t,dt in progbar(chrono.ticker):
      # Forecast
      mu = Dyn(mu,t-dt,dt)
      P  = 2*PC*WaveC(k)

      if kObs is not None:
        stats.assess(k,kObs,'f',mu=mu,Cov=P)
        # Analysis
        P *= infl
        H  = Obs.jacob(mu,t)
        KG = mrdiv(P@H.T, H@P@H.T + Obs.noise.C.full)
        KH = KG@H
        mu = mu + KG@(yy[kObs] - Obs(mu,t))

        # Re-calibrate wave_crest with new W0 = Pa/(2*PC).
        # Note: obs innovations are not used to estimate P!
        Pa    = (eye(Dyn.M) - KH) @ P
        WaveC = wave_crest(trace(Pa)/trace(2*PC),CorrL)

      stats.assess(k,kObs,mu=mu,Cov=2*PC*WaveC(k,kObs))
  return assimilator


def wave_crest(W0,L):
  """Return a sigmoid [function W(k)] that may be used
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





