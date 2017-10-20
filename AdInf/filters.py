from common import *

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



