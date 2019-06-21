"""The extended KF (EKF) and the (Rauch-Tung-Striebel) smoother."""

from dapper import *

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
  def assimilator(stats,HMM,xx,yy):
    Dyn,Obs,chrono,X0 = HMM.Dyn, HMM.Obs, HMM.t, HMM.X0

    R  = Obs.noise.C.full
    Q  = 0 if Dyn.noise.C==0 else Dyn.noise.C.full

    mu = X0.mu
    P  = X0.C.full

    stats.assess(0,mu=mu,Cov=P)

    for k,kObs,t,dt in progbar(chrono.ticker):
      
      mu = Dyn(mu,t-dt,dt)
      F  = Dyn.jacob(mu,t-dt,dt) 
      P  = infl**(dt)*(F@P@F.T) + dt*Q

      # Of academic interest? Higher-order linearization:
      # mu_i += 0.5 * (Hessian[f_i] * P).sum()

      if kObs is not None:
        stats.assess(k,kObs,'f',mu=mu,Cov=P)
        H  = Obs.jacob(mu,t)
        KG = mrdiv(P @ H.T, H@P@H.T + R)
        y  = yy[kObs]
        mu = mu + KG@(y - Obs(mu,t))
        KH = KG@H
        P  = (eye(Dyn.M) - KH) @ P

        stats.trHK[kObs] = trace(KH)/Dyn.M

      stats.assess(k,kObs,mu=mu,Cov=P)
  return assimilator


# TODO: Clean up
@DA_Config
def ExtRTS(infl=1.0,**kwargs):
  """
  """
  def assimilator(stats,HMM,xx,yy):
    Dyn,Obs,chrono,X0 = HMM.Dyn, HMM.Obs, HMM.t, HMM.X0
    Nx = Dyn.M

    R  = Obs.noise.C.full
    Q  = 0 if Dyn.noise.C==0 else Dyn.noise.C.full

    mu    = zeros((chrono.K+1,Nx))
    P     = zeros((chrono.K+1,Nx,Nx))

    # Forecasted values
    muf   = zeros((chrono.K+1,Nx))
    Pf    = zeros((chrono.K+1,Nx,Nx))
    Ff    = zeros((chrono.K+1,Nx,Nx))

    mu[0] = X0.mu
    P [0] = X0.C.full

    stats.assess(0,mu=mu[0],Cov=P[0])

    # Forward pass
    for k,kObs,t,dt in progbar(chrono.ticker, 'ExtRTS->'):
      mu[k]  = Dyn(mu[k-1],t-dt,dt)
      F      = Dyn.jacob(mu[k-1],t-dt,dt) 
      P [k]  = infl**(dt)*(F@P[k-1]@F.T) + dt*Q

      # Store forecast and Jacobian
      muf[k] = mu[k]
      Pf [k] = P [k]
      Ff [k] = F

      if kObs is not None:
        stats.assess(k,kObs,'f',mu=mu[k],Cov=P[k])
        H     = Obs.jacob(mu[k],t)
        KG    = mrdiv(P[k] @ H.T, H@P[k]@H.T + R)
        y     = yy[kObs]
        mu[k] = mu[k] + KG@(y - Obs(mu[k],t))
        KH    = KG@H
        P[k]  = (eye(Nx) - KH) @ P[k]
        stats.assess(k,kObs,'a',mu=mu[k],Cov=P[k])

    # Backward pass
    for k in progbar(range(chrono.K)[::-1],'ExtRTS<-'):
      J     = mrdiv(P[k]@Ff[k+1].T, Pf[k+1])
      mu[k] = mu[k]  + J @ (mu[k+1]  - muf[k+1])
      P[k]  = P[k] + J @ (P[k+1] - Pf[k+1]) @ J.T
    for k in progbar(range(chrono.K+1),desc='Assess'):
      stats.assess(k,mu=mu[k],Cov=P[k])

  return assimilator



