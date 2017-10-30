from common import *

@DA_Config
def EAKF_A07_v1(N,v_b=None,damp=0.9,damp_mu=1,CLIP=0.8,ordr='rand',infl=1.0,rot=False,**kwargs):
  """
  Damping is very important!

  v_b does not CV to 0 (when using estimate_v_b).
  This does not seem to be due to min(0.99999,Rtio),
  but rather the fitting approach(?).

  Bug test: set v_b really small (1e-6) to disable adaptive inflation,
            and check that it reproduces scores of the EnKF('Sqrt')
            with fixed/tuned infl.
  """
  def assimilator(stats,twin,xx,yy):
    f,h,chrono,X0 = twin.f, twin.h, twin.t, twin.X0

    n = N-1

    R    = h.noise
    Rm12 = h.noise.C.sym_sqrt_inv

    E = X0.sample(N)
    stats.assess(0,E=E)

    stats.infl_mean = zeros(chrono.KObs+1)
    stats.v_b       = zeros(chrono.KObs+1)

    l2_b = 1.0 # mean of prior inflation
    nonlocal v_b
    if v_b:
      estimate_v_b = False
    else: 
      estimate_v_b = True
      v_b = 10 # init

    for k,kObs,t,dt in progbar(chrono.forecast_range):
      E = f(E,t-dt,dt)
      E = add_noise(E, dt, f.noise, kwargs)

      if kObs is not None:
        stats.assess(k,kObs,'f',E=E)
        y    = yy[kObs]
        inds = serial_inds(ordr, y, R, anom(E)[0])
            
        inflations = zeros(h.m)
        for i,j in enumerate(inds):

          l1 = 1.0
          for UPDATE_TARGET in ['INFL','STATE']:
            E  = post_process(E,l1,0)

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

            if UPDATE_TARGET=='INFL':
              D     = dyj   # observed mean innovation
              so    = 1     # "sigma_o^2" = 1 coz of normalizt. above
              sp    = skk/n # "sigma_p^2"
              T2    = lambda l2: l2*sp + so # theta^2
              zz    = np.roots([1, -T2(l2_b), .5*v_b*sp**2, -.5*v_b*(sp*D)**2])
              zz    = zz[np.isreal(zz)].real
              l2s   = (zz-so)/sp # Solve z = T2(lambda)
              idx   = (np.abs(zz-l2_b)).argmin() # find closest to l2_b
              l2u   = l2s[idx] # posterior mean
              l2_c  = damp_mu**(2/h.m) if damp_mu!='b' else l2_b
              l2u   = l2_c + damp*(l2u-l2_c)
              l2u   = max(CLIP,l2u)

              # Fit posterior var via ratio of posterior eval'd at pt1/pt2
              if estimate_v_b: 
                pt1   = l2u+sqrt(v_b)
                pt2   = l2u
                Rtio  = exp(( (pt2-l2_b)**2 - (pt1-l2_b)**2 )/(2*v_b)) # prior
                Rtio *= sqrt(T2(pt2)/T2(pt1))                 # lklhd determinant
                Rtio *= exp( D**2/2*(1/T2(pt2) - 1/T2(pt1)) ) # lklhd exponent
                Rtio  = min(0.9999,Rtio)
                v_b   = -v_b / 2 / log(Rtio)

              l2_b  = l2u # new prior mean
              l1    = sqrt(l2u)
              inflations[i] = l1

            elif UPDATE_TARGET=='STATE':
              if skk<1e-9: continue

              # Update state (regress from y2)
              Regression = A.T @ Yj/np.sum(Yj**2)
              mu        += Regression*dy2
              A         += np.outer(Y2 - Yj, Regression)
              E = mu + A

            else: raise Exception

        stats.infl     [kObs] = inflations.prod()
        stats.infl_mean[kObs] = exp(mean(log(inflations)))
        stats.v_b      [kObs] = v_b
        E = post_process(E,infl,rot)
      stats.assess(k,kObs,E=E)
  return assimilator


@DA_Config
def ETKF_M11_v1(N,v_b,v_o=None,CLIP=0.8,infl=1.0,rot=False,**kwargs):
  """
  M11 also uses 1.5% fixed inflation.

  Bug test: set v_b really small (1e-6) to disable adaptive inflation,
            and check that it reproduces scores of the EnKF('Sqrt')
            with fixed/tuned infl.
  """
    
  def assimilator(stats,twin,xx,yy):
    f,h,chrono,X0,R  = twin.f, twin.h, twin.t, twin.X0, twin.h.noise.C

    # Init
    E = X0.sample(N)
    stats.assess(0,E=E)

    nonlocal v_o
    estimate_v_o = True if not v_o else False
    stats.v_o = zeros(chrono.KObs+1)

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

        trHPHR  = trace(YR.T @ YR)/(N-1)
        infl2_o = (dR@dR - h.m)/trHPHR
        infl2_o = max(CLIP,infl2_o)
        infl2_b = infl2_a if 'infl2_a' in locals() else 1.0
        if estimate_v_o:
          v_o = 2/h.m * ( (infl2_b*trHPHR+h.m)/trHPHR )**2
          stats.v_o[kObs] = v_o
        infl2_a = (infl2_b/v_b + infl2_o/v_o)/(1/v_b + 1/v_o)
        stats.infl[kObs] = sqrt(infl2_a)

        w = dy @ R.inv @ Y.T @ Pw # infl not used here
        E = mu + w@A + (sqrt(infl2_a)*T)@A

        E = post_process(E,infl,rot)
      stats.assess(k,kObs,E=E)
  return assimilator













# THOUGHTS:
# Remember that the KF is the optimal solution in the LG case.
# I.e. it cannot be imporoved by adaptive inflation estimation
# (assuming no model error).
# Similarly, a well-tuned, static-background 3D-Var is optimal in its own way,
# and so cannot be improved on, except if there is significant NonLinearity.

from scipy.integrate import quad
from scipy.optimize import minimize_scalar as minz

def Chi2_pdf(d,nu,t):
  # c = same as for iChi2, I believe.
  c = 1 # normalize outside.
  return c * 1/t * (d/t)**(nu/2-1) * exp(-nu*d/2/t)

def iChi2_pdf(s,nu,x):
  c = 1 # normalize outside.
  #c = nu**(nu/2) / 2**(nu/2) / sp.special.gamma(nu/2) * s**(nu/2)
  return c * x**(-nu/2-1) * exp(-nu*s/2/x)

# IF     f = lambda x: iChi2_pdf(s,nu, x) # Use iChi2_pdf normalized
# THEN   ( f(md+eps)-f(md) ) - ( f(md)-f(md-eps) ) / eps**2 # Hessian
# EQUALS -exp(-nu/2-1) / 2**(nu/2+2) / (s*nu)**3 ...        # found by
#        ... * (nu+2)**(nu/2+4) / sp.special.gamma(nu/2)    # WolfAlph

class InvChi2Filter(MLR_Print):
  def __init__(self,s=1.0,nu=5,L=None):
    """
    Start at nu=5 so that the posterior is sure to have a variance
    (which we use for comparison).
    """
    if L is None:
      self.forget = exp(-1/corr_L(xx))
    else:
      self.forget = exp(-1/L)
    self.nu = max(1e-4,nu)
    self.s  = s
  def forecast(self,k=1):
    self.nu *= self.forget**k
    #self.s  = 1.0 + self.forget**k*(self.s - 1.0)
  def update(self,lklhd):
    Domain  = (1e-10, min(self.s*100, 1e-5**(-2/self.nu)))
    quad_   = lambda f: quad(f,*Domain)[0]
    prior   = lambda x: iChi2_pdf(self.s,self.nu,x)
    post_nn = lambda x: prior(x) * lklhd(x)
    normlzt = quad_(lambda x:  post_nn(x))
    post    = lambda x: post_nn(x) / normlzt
    #
    mean    = quad_(lambda x:  post(x)*x)
    var     = quad_(lambda x:  post(x)*(x-mean)**2)
    self.nu = 4 + 2*mean**2/var
    self.s  = (self.nu-2)/self.nu * mean
    #
    # Using mode instead of mean (untested):
    #mode    = minz(lambda x: -post(x),              Domain).x
    #polynom = [var, -8*var-2*mode**2, 20*var-8*mode**2, -16*var-8*mode**2]
    #roots   = np.roots(polynom)
    #nu      = np.real(roots[np.isreal(roots)])
    #if len(nu)>1:
      #raise raise_AFE("Found more than 1 root")
    #else:
      #self.nu = nu[0]
    #self.s = mode*
    #
    # Do as Anderson'2009: Use ratio mode/mode+1
    





@DA_Config
def Var3D_A(fs=1,infl=1.0,**kwargs):
  """Only estimate, don't use, the InvChi2Filter inflation."""
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

    InFl = InvChi2Filter(s=1.0,nu=5,L=CorrL/fs)

    for k,kObs,t,dt in progbar(chrono.forecast_range):
      # Forecast
      mu = f(mu,t-dt,dt)
      P  = 2*PC*WaveC(k)
      InFl.forecast()

      if kObs is not None:


        P_ = InFl.s * 3*P
        #
        # Scaling estimation
        H   = h.jacob(mu,t)
        R   = h.noise.C.full
        d   = yy[kObs] - h(mu,t)
        tR  = trace(R)
        tP  = trace(H@P_@H.T)
        rho = tP/tR
        #
        U   = CovMat(H@P_@H.T + R).sym_sqrt_inv
        dz  = U @ d
        Dlt = (dz**2).sum()/h.m # expected value: 1
        #
        theta = lambda x: (x*rho+1)/(rho+1)
        lklhd = lambda x: Chi2_pdf(Dlt, h.m, theta(x))
        #
        InFl.update(lklhd)
        stats.infl[kObs] = InFl.s


        stats.assess(k,kObs,'f',mu=mu,Cov=P)
        # Analysis
        P *= infl
        H  = h.jacob(mu,t)
        KG = mrdiv(P@H.T, H@P@H.T + h.noise.C.full)
        KH = KG@H
        mu = mu + KG@(yy[kObs] - h(mu,t))

        # Re-calibrate wave_crest with new W0 = Pa/(2*PC)
        # Note: obs innovations are not used to estimate P!
        Pa    = (eye(f.m) - KH) @ P
        WaveC = wave_crest(trace(Pa)/trace(2*PC),CorrL)

      stats.assess(k,kObs,mu=mu,Cov=2*PC*WaveC(k,kObs))
  return assimilator



@DA_Config
def Var3D_B(fs=1,infl=1.0,**kwargs):
  """Uses InvChi2Filter, with a climatologial reference."""
  def assimilator(stats,twin,xx,yy):
    f,h,chrono,X0 = twin.f, twin.h, twin.t, twin.X0

    # Compute "climatology"
    muC = mean(xx,0)
    AC  = xx - muC
    PC  = (AC.T @ AC)/(xx.shape[0] - 1)

    # Init
    mu = muC
    P  = PC
    InFl = InvChi2Filter(s=1.0,nu=5,L=CorrL/fs)
    stats.assess(0,mu=mu,Cov=P)


    for k,kObs,t,dt in progbar(chrono.forecast_range):
      # Forecast
      mu = f(mu,t-dt,dt)
      InFl.forecast()

      if kObs is not None:

        P_ = 0.3*PC # Reference variance
        #
        # Scaling estimation
        H   = h.jacob(mu,t)
        R   = h.noise.C.full
        d   = yy[kObs] - h(mu,t)
        tR  = trace(R)
        tP  = trace(H@P_@H.T)
        rho = tP/tR
        #
        U   = CovMat(H@P_@H.T + R).sym_sqrt_inv
        dz  = U @ d
        Dlt = (dz**2).sum()/h.m # expected value: 1
        #
        theta = lambda x: (x*rho+1)/(rho+1)
        lklhd = lambda x: Chi2_pdf(Dlt, h.m, theta(x))
        #
        InFl.update(lklhd)
        stats.infl[kObs] = InFl.s
        P = InFl.s*infl*P_


        stats.assess(k,kObs,'f',mu=mu,Cov=P)
        # Analysis
        H  = h.jacob(mu,t)
        KG = mrdiv(P@H.T, H@P@H.T + h.noise.C.full)
        KH = KG@H
        mu = mu + KG@(yy[kObs] - h(mu,t))
        Pa = (eye(f.m) - KH) @ P
        P  = Pa

      stats.assess(k,kObs,mu=mu,Cov=P)
  return assimilator



