"""Variational DA methods (iEnKS, 4D-Var, etc)"""

from typing import Optional

import numpy as np
import scipy.linalg as sla

from dapper.da_methods.ensemble import hyperprior_coeffs, post_process, zeta_a
from dapper.stats import center, inflate_ens, mean0
from dapper.tools.linalg import pad0, svd0, tinv
from dapper.tools.matrices import CovMat
from dapper.tools.progressbar import progbar

from . import da_method


@da_method
class var_method:
    "Declare default variational arguments."
    Lag: int    = 1
    nIter: int  = 10
    wtol: float = 0


@var_method
class iEnKS:
    """Iterative EnKS.

    Special cases: EnRML, ES-MDA, iEnKF, EnKF `bib.raanes2019revising`.

    As in `bib.bocquet2014iterative`, optimization uses Gauss-Newton.
    See `bib.bocquet2012combining` for Levenberg-Marquardt.
    If MDA=True, then there's not really any optimization,
    but rather Gaussian annealing.

    Args:
      upd_a (str):
        Analysis update form (flavour). One of:

        - "Sqrt"   : as in ETKF  , using a deterministic matrix square root transform.
        - "PertObs": as in EnRML , using stochastic, perturbed-observations.
        - "Order1" : as in DEnKF of `bib.sakov2008deterministic`.

      Lag:
        Length of the DA window (DAW), in multiples of dkObs (i.e. cycles).

        - Lag=1 (default) => iterative "filter" iEnKF `bib.sakov2012iterative`.
        - Lag=0           => maximum-likelihood filter `bib.zupanski2005maximum`.

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

    Refs: `bib.bocquet2012combining`, `bib.bocquet2013joint`,
    `bib.bocquet2014iterative`.
    """
    upd_a: str
    N: int
    MDA: bool    = False
    step: bool   = False
    bundle: bool = False
    xN: float    = None
    infl: float  = 1.0
    rot: bool    = False

    # NB It's very difficult to preview what should happen to
    # all of the time indices in all cases of nIter and Lag.
    # => Any changes to this function must be unit-tested via
    # scripts/test_iEnKS.py.

    # TODO 6:
    # - step length
    # - Implement quasi-static assimilation. Boc notes:
    #   * The 'balancing step' is complicated.
    #   * Trouble playing nice with '-N' inflation estimation.

    def assimilate(self, HMM, xx, yy):
        Dyn, Obs, chrono, X0, stats, N = \
            HMM.Dyn, HMM.Obs, HMM.t, HMM.X0, self.stats, self.N
        R, KObs, N1 = HMM.Obs.noise.C, HMM.t.KObs, N-1
        Rm12 = R.sym_sqrt_inv

        assert Dyn.noise.C == 0, (
            "Q>0 not yet supported."
            " See Sakov et al 2017: 'An iEnKF with mod. error'")

        if self.bundle:
            EPS = 1e-4  # Sakov/Boc use T=EPS*eye(N), with EPS=1e-4, but I ...
        else:
            EPS = 1.0  # ... prefer using  T=EPS*T, yielding a conditional cloud shape

        # Initial ensemble
        E = X0.sample(N)

        # Loop over DA windows (DAW).
        for kObs in progbar(np.arange(-1, KObs+self.Lag+1)):
            kLag = kObs-self.Lag
            DAW  = range(max(0, kLag+1), min(kObs, KObs) + 1)

            # Assimilation (if ∃ "not-fully-assimlated" obs).
            if 0 <= kObs <= KObs:

                # Init iterations.
                X0, x0 = center(E)    # Decompose ensemble.
                w      = np.zeros(N)  # Control vector for the mean state.
                T      = np.eye(N)    # Anomalies transform matrix.
                Tinv   = np.eye(N)
                # Explicit Tinv [instead of tinv(T)] allows for merging MDA code
                # with iEnKS/EnRML code, and flop savings in 'Sqrt' case.

                for iteration in np.arange(self.nIter):
                    # Reconstruct smoothed ensemble.
                    E = x0 + (w + EPS*T)@X0
                    # Forecast.
                    for kCycle in DAW:
                        for k, t, dt in chrono.cycle(kCycle):
                            E = Dyn(E, t-dt, dt)
                    # Observe.
                    Eo = Obs(E, t)

                    # Undo the bundle scaling of ensemble.
                    if EPS != 1.0:
                        E  = inflate_ens(E, 1/EPS)
                        Eo = inflate_ens(Eo, 1/EPS)

                    # Assess forecast stats; store {Xf, T_old} for analysis assessment.
                    if iteration == 0:
                        stats.assess(k, kObs, 'f', E=E)
                        Xf, xf = center(E)
                    T_old = T

                    # Prepare analysis.
                    y      = yy[kObs]           # Get current obs.
                    Y, xo  = center(Eo)         # Get obs {anomalies, mean}.
                    dy     = (y - xo) @ Rm12.T  # Transform obs space.
                    Y      = Y        @ Rm12.T  # Transform obs space.
                    Y0     = Tinv @ Y           # "De-condition" the obs anomalies.
                    V, s, UT = svd0(Y0)         # Decompose Y0.

                    # Set "cov normlzt fctr" za ("effective ensemble size")
                    # => pre_infl^2 = (N-1)/za.
                    if self.xN is None:
                        za  = N1
                    else:
                        za  = zeta_a(*hyperprior_coeffs(s, N, self.xN), w)
                    if self.MDA:
                        # inflation (factor: nIter) of the ObsErrCov.
                        za *= self.nIter

                    # Post. cov (approx) of w,
                    # estimated at current iteration, raised to power.
                    def Cowp(expo): return (V * (pad0(s**2, N) + za)**-expo) @ V.T
                    Cow1 = Cowp(1.0)

                    if self.MDA:  # View update as annealing (progressive assimilation).
                        Cow1 = Cow1 @ T  # apply previous update
                        dw = dy @ Y.T @ Cow1
                        if 'PertObs' in self.upd_a:   # == "ES-MDA". By Emerick/Reynolds
                            D   = mean0(np.random.randn(*Y.shape)) * np.sqrt(self.nIter)
                            T  -= (Y + D) @ Y.T @ Cow1
                        elif 'Sqrt' in self.upd_a:    # == "ETKF-ish". By Raanes
                            T   = Cowp(0.5) * np.sqrt(za) @ T
                        elif 'Order1' in self.upd_a:  # == "DEnKF-ish". By Emerick
                            T  -= 0.5 * Y @ Y.T @ Cow1
                        # Tinv = eye(N) [as initialized] coz MDA does not de-condition.

                    else:  # View update as Gauss-Newton optimzt. of log-posterior.
                        grad  = Y0@dy - w*za                  # Cost function gradient
                        dw    = grad@Cow1                     # Gauss-Newton step
                        # ETKF-ish". By Bocquet/Sakov.
                        if 'Sqrt' in self.upd_a:
                            # Sqrt-transforms
                            T     = Cowp(0.5) * np.sqrt(N1)
                            Tinv  = Cowp(-.5) / np.sqrt(N1)
                            # Tinv saves time [vs tinv(T)] when Nx<N
                        # "EnRML". By Oliver/Chen/Raanes/Evensen/Stordal.
                        elif 'PertObs' in self.upd_a:
                            D     = mean0(np.random.randn(*Y.shape)) \
                                if iteration == 0 else D
                            gradT = -(Y+D)@Y0.T + N1*(np.eye(N) - T)
                            T     = T + gradT@Cow1
                            # Tinv= tinv(T, threshold=N1)  # unstable
                            Tinv  = sla.inv(T+1)           # the +1 is for stability.
                        # "DEnKF-ish". By Raanes.
                        elif 'Order1' in self.upd_a:
                            # Included for completeness; does not make much sense.
                            gradT = -0.5*Y@Y0.T + N1*(np.eye(N) - T)
                            T     = T + gradT@Cow1
                            Tinv  = tinv(T, threshold=N1)

                    w += dw
                    if dw@dw < self.wtol*N:
                        break

                # Assess (analysis) stats.
                # The final_increment is a linearization to
                # (i) avoid re-running the model and
                # (ii) reproduce EnKF in case nIter==1.
                final_increment = (dw+T-T_old)@Xf
                # See docs/snippets/iEnKS_Ea.jpg.
                stats.assess(k, kObs, 'a', E=E+final_increment)
                stats.iters[kObs] = iteration+1
                if self.xN:
                    stats.infl[kObs] = np.sqrt(N1/za)

                # Final (smoothed) estimate of E at [kLag].
                E = x0 + (w+T)@X0
                E = post_process(E, self.infl, self.rot)

            # Slide/shift DAW by propagating smoothed ('s') ensemble from [kLag].
            if -1 <= kLag < KObs:
                if kLag >= 0:
                    stats.assess(chrono.kkObs[kLag], kLag, 's', E=E)
                for k, t, dt in chrono.cycle(kLag+1):
                    stats.assess(k-1, None, 'u', E=E)
                    E = Dyn(E, t-dt, dt)

        stats.assess(k, KObs, 'us', E=E)


@var_method
class Var4D:
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
    """
    B: Optional[np.ndarray] = None
    xB: float               = 1.0

    def assimilate(self, HMM, xx, yy):
        Dyn, Obs, chrono, X0, stats = HMM.Dyn, HMM.Obs, HMM.t, HMM.X0, self.stats
        R, KObs = HMM.Obs.noise.C, HMM.t.KObs
        Rm12 = R.sym_sqrt_inv
        Nx = Dyn.M

        # Set background covariance. Note that it is static (compare to iEnKS).
        if self.B in (None, 'clim'):
            # Use climatological cov, ...
            B = np.cov(xx.T)  # ... estimated from truth
        elif self.B == 'eye':
            B = np.eye(Nx)
        else:
            B = self.B
        B *= self.xB
        B12 = CovMat(B).sym_sqrt

        # Init
        x = X0.mu
        stats.assess(0, mu=x, Cov=B)

        # Loop over DA windows (DAW).
        for kObs in progbar(np.arange(-1, KObs+self.Lag+1)):
            kLag = kObs-self.Lag
            DAW = range(max(0, kLag+1), min(kObs, KObs) + 1)

            # Assimilation (if ∃ "not-fully-assimlated" obs).
            if 0 <= kObs <= KObs:

                # Init iterations.
                w   = np.zeros(Nx)  # Control vector for the mean state.
                x0  = x.copy()     # Increment reference.

                for iteration in np.arange(self.nIter):
                    # Reconstruct smoothed state.
                    x = x0 + B12@w
                    X = B12  # Aggregate composite TLMs onto B12
                    # Forecast.
                    for kCycle in DAW:
                        for k, t, dt in chrono.cycle(kCycle):
                            X = Dyn.linear(x, t-dt, dt) @ X
                            x = Dyn(x, t-dt, dt)

                    # Assess forecast stats
                    if iteration == 0:
                        stats.assess(k, kObs, 'f', mu=x, Cov=X@X.T)

                    # Observe.
                    Y  = Obs.linear(x, t) @ X
                    xo = Obs(x, t)

                    # Analysis prep.
                    y      = yy[kObs]          # Get current obs.
                    dy     = Rm12 @ (y - xo)   # Transform obs space.
                    Y      = Rm12 @ Y          # Transform obs space.
                    V, s, UT = svd0(Y.T)         # Decomp for lin-alg update comps.

                    # Post. cov (approx) of w,
                    # estimated at current iteration, raised to power.
                    Cow1 = (V * (pad0(s**2, Nx) + 1)**-1.0) @ V.T

                    # Compute analysis update.
                    grad = Y.T@dy - w          # Cost function gradient
                    dw   = Cow1@grad           # Gauss-Newton step
                    w   += dw                  # Step

                    if dw@dw < self.wtol*Nx:
                        break

                # Assess (analysis) stats.
                final_increment = X@dw
                stats.assess(k,   kObs, 'a', mu=x+final_increment, Cov=X@Cow1@X.T)
                stats.iters[kObs] = iteration+1

                # Final (smoothed) estimate at [kLag].
                x = x0 + B12@w
                X = B12

            # Slide/shift DAW by propagating smoothed ('s') state from [kLag].
            if -1 <= kLag < KObs:
                if kLag >= 0:
                    stats.assess(chrono.kkObs[kLag], kLag, 's', mu=x, Cov=X@Cow1@X.T)
                for k, t, dt in chrono.cycle(kLag+1):
                    stats.assess(k-1, None, 'u', mu=x, Cov=Y@Y.T)
                    X = Dyn.linear(x, t-dt, dt) @ X
                    x = Dyn(x, t-dt, dt)

        stats.assess(k, KObs, 'us', mu=x, Cov=X@Cow1@X.T)
