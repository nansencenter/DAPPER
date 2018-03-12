####################################################
# Copied some methods from da_methods.py.
# Make them use only diag(R) !!!
# Refer to da_methods.py for documentation on methods.
####################################################

from common import *

@DA_Config
def EnKF_N_diagR(N,infl=1.0,rot=False,Hess=False,**kwargs):
  def assimilator(stats,twin,xx,yy):
    # Unpack
    f,h,chrono,X0  = twin.f, twin.h, twin.t, twin.X0

    diagR = diag(h.noise.C.full)
    Rm12  = diag(diagR**(-0.5))
    Ri    = diag(diagR**(-1.0))

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
def DEnKF_diagR(N,infl=1.0,rot=False,**kwargs):
  def assimilator(stats,twin,xx,yy):
    f,h,chrono,X0 = twin.f, twin.h, twin.t, twin.X0

    R = diag(diag(h.noise.C.full))

    E = X0.sample(N)
    stats.assess(0,E=E)

    for k,kObs,t,dt in progbar(chrono.forecast_range):
      E  = f.model(E,t-dt,dt)
      E += sqrt(dt)*f.noise.sample(N)

      if kObs is not None:
        stats.assess(k,kObs,'f',E=E)

        hE = h.model(E,t)
        y  = yy[kObs]

        mu = mean(E,0)
        A  = E - mu

        hx = mean(hE,0)
        Y  = hE-hx
        dy = y - hx

        C  = Y.T @ Y + R*(N-1)
        YC = mrdiv(Y, C)
        KG = A.T @ YC
        E  = E + KG@dy - 0.5*(KG@Y.T).T

        E = post_process(E,infl,rot)

      stats.assess(k,kObs,E=E)
  return assimilator

