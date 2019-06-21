"""More experimental or esoteric DA methods."""

from dapper import *

@DA_Config
def RHF(N,ordr='rand',infl=1.0,rot=False,**kwargs):
  """Rank histogram filter [And10]_.

  Quick & dirty implementation without attention to (de)tails.

  Settings for reproducing literature benchmarks may be found in
  :mod:`dapper.mods.Lorenz63.anderson2010non` 
  """
  def assimilator(stats,HMM,xx,yy):
    Dyn,Obs,chrono,X0 = HMM.Dyn, HMM.Obs, HMM.t, HMM.X0

    N1         = N-1
    step       = 1/N
    cdf_grid   = linspace(step/2, 1-step/2, N)

    R    = Obs.noise
    Rm12 = Obs.noise.C.sym_sqrt_inv

    E = X0.sample(N)
    stats.assess(0,E=E)

    for k,kObs,t,dt in progbar(chrono.ticker):
      E = Dyn(E,t-dt,dt)
      E = add_noise(E, dt, Dyn.noise, kwargs)

      if kObs is not None:
        stats.assess(k,kObs,'f',E=E)
        y    = yy[kObs]
        inds = serial_inds(ordr, y, R, center(E)[0])
            
        for i,j in enumerate(inds):
          Eo = Obs(E,t)
          xo = mean(Eo,0)
          Y  = Eo - xo
          mu = mean(E ,0)
          A  = E-mu

          # Update j-th component of observed ensemble
          dYf    = Rm12[j,:] @ (y - Eo).T # NB: does Rm12 make sense?
          Yj     = Rm12[j,:] @ Y.T
          Regr   = A.T@Yj/np.sum(Yj**2)

          Sorted = np.argsort(dYf)
          Revert = np.argsort(Sorted)
          dYf    = dYf[Sorted]
          w      = reweight(ones(N),innovs=dYf[:,None]) # Lklhd
          w      = w.clip(1e-10) # Avoid zeros in interp1
          cw     = w.cumsum()
          cw    /= cw[-1]
          cw    *= N1/N
          cdfs   = np.minimum(np.maximum(cw[0],cdf_grid),cw[-1])
          dhE    = -dYf + np.interp(cdfs, cw, dYf)
          dhE    = dhE[Revert]
          # Update state by regression
          E     += np.outer(-dhE, Regr)

        E = post_process(E,infl,rot)

      stats.assess(k,kObs,E=E)
  return assimilator




@DA_Config
def LNETF(N,loc_rad,taper='GC',infl=1.0,Rs=1.0,rot=False,**kwargs):
  """The Nonlinear-Ensemble-Transform-Filter (localized) [Wil16]_, [Töd15]_.

  It is (supposedly) a deterministic upgrade of the NLEAF of [Lei11]_.

  Settings for reproducing literature benchmarks may be found in

  - :mod:`dapper.mods.Lorenz95.tod15`
  - :mod:`dapper.mods.Lorenz95.wiljes2017`
  """
  def assimilator(stats,HMM,xx,yy):
    Dyn,Obs,chrono,X0 = HMM.Dyn, HMM.Obs, HMM.t, HMM.X0
    Rm12 = Obs.noise.C.sym_sqrt_inv

    E = X0.sample(N)
    stats.assess(0,E=E)

    for k,kObs,t,dt in progbar(chrono.ticker):
      E = Dyn(E,t-dt,dt)
      E = add_noise(E, dt, Dyn.noise, kwargs)

      if kObs is not None:
        stats.assess(k,kObs,'f',E=E)
        mu = mean(E,0)
        A  = E - mu

        Eo = Obs(E,t)
        xo = mean(Eo,0)
        YR = (Eo-xo)  @ Rm12.T
        yR = (yy[kObs] - xo) @ Rm12.T

        state_batches, obs_taperer = Obs.localizer(loc_rad, 'x2y', t, taper)
        for ii in state_batches:
          # Localize obs
          jj, tapering = obs_taperer(ii)
          if len(jj) == 0: return

          Y_jj  = YR[:,jj] * sqrt(tapering)
          dy_jj = yR[jj]   * sqrt(tapering)

          # NETF:
          # This "paragraph" is the only difference to the LETKF.
          innovs = (dy_jj-Y_jj)/Rs
          if 'laplace' in str(type(Obs.noise)).lower():
            w    = laplace_lklhd(innovs)
          else: # assume Gaussian
            w    = reweight(ones(N),innovs=innovs)
          dmu    = w@A[:,ii]
          AT     = sqrt(N)*funm_psd(diag(w) - np.outer(w,w), sqrt)@A[:,ii]

          E[:,ii] = mu[ii] + dmu + AT

        E = post_process(E,infl,rot)
      stats.assess(k,kObs,E=E)
  return assimilator

def laplace_lklhd(xx):
  """Compute a Laplacian likelihood.

  Compute likelihood of xx wrt. the sampling distribution
  LaplaceParallelRV(C=I), i.e., for x in xx:
  p(x) = exp(-sqrt(2)*|x|_1) / sqrt(2).
  """
  logw   = -sqrt(2)*np.sum(np.abs(xx), axis=1)
  logw  -= logw.max()    # Avoid numerical error
  w      = exp(logw)     # non-log
  w     /= w.sum()       # normalize
  return w


