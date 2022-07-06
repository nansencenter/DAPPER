"""Variational DA methods (iEnKS, 4D-Var, etc)."""

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
    """Declare default variational arguments."""

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
        Length of the DA window (DAW), in multiples of dko (i.e. cycles).

        - Lag=1 (default) => iterative "filter" iEnKF `bib.sakov2012iterative`.
        - Lag=0           => maximum-likelihood filter `bib.zupanski2005maximum`.

      Shift : How far (in cycles) to slide the DAW.
              Fixed at 1 for code simplicity.

      nIter : Maximal num. of iterations used (>=1). Default: 10.
              Supporting nIter==0 requires more code than it's worth.

      wtol  : Rel. tolerance defining convergence.
              Default: 0 => always do nIter iterations.
              Recommended: 1e-5.

      MDA   : Use iterations of the "multiple data assimlation" type.
              Ref `bib.emerick2012history`

      bundle: Use finite-diff. linearization instead of of least-squares regression.
              Makes the iEnKS very much alike the iterative, extended KF (IEKS).

      xN    : If set, use EnKF_N() pre-inflation. See further documentation there.

    Total number of model simulations (of duration dto): N * (nIter*Lag + 1).
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
        R, Ko  = HMM.Obs.noise.C, HMM.tseq.Ko
        Rm12 = R.sym_sqrt_inv

        assert HMM.Dyn.noise.C == 0, (
            "Q>0 not yet supported."
            " See Sakov et al 2017: 'An iEnKF with mod. error'")

        if self.bundle:
            EPS = 1e-4  # Sakov/Boc use T=EPS*eye(N), with EPS=1e-4, but I ...
        else:
            EPS = 1.0  # ... prefer using  T=EPS*T, yielding a conditional cloud shape

        # Initial ensemble
        E = HMM.X0.sample(self.N)

        # Forward ensemble to ko = 0 if Lag = 0
        t = 0
        k = 0
        if self.Lag == 0:
            for k, t, dt in HMM.tseq.cycle(ko=0):
                self.stats.assess(k-1, None, 'u', E=E)
                E = HMM.Dyn(E, t-dt, dt)

        # Loop over DA windows (DAW).
        for ko in progbar(range(0, Ko+self.Lag+1)):
            kLag = ko-self.Lag
            DAW  = range(max(0, kLag+1), min(ko, Ko) + 1)

            # Assimilation (if ∃ "not-fully-assimlated" Obs).
            if ko <= Ko:
                E = iEnKS_update(self.upd_a, E, DAW, HMM, self.stats,
                                 EPS, yy[ko], (k, ko, t), Rm12,
                                 self.xN, self.MDA, (self.nIter, self.wtol))
                E = post_process(E, self.infl, self.rot)

            # Slide/shift DAW by propagating smoothed ('s') ensemble from [kLag].
            if kLag >= 0:
                self.stats.assess(HMM.tseq.kko[kLag], kLag, 's', E=E)
            cycle_window = range(max(kLag+1, 0), min(max(kLag+1+1, 0), Ko+1))

            for kCycle in cycle_window:
                for k, t, dt in HMM.tseq.cycle(kCycle):
                    self.stats.assess(k-1, None, 'u', E=E)
                    E = HMM.Dyn(E, t-dt, dt)

        self.stats.assess(k, Ko, 'us', E=E)


def iEnKS_update(upd_a, E, DAW, HMM, stats, EPS, y, time, Rm12, xN, MDA, threshold):
    """Perform the iEnKS update.

    This implementation includes several flavours and forms,
    specified by `upd_a` (See `iEnKS`)
    """
    # distribute variable
    k, ko, t = time
    nIter, wtol = threshold
    N, Nx = E.shape

    # Init iterations.
    N1 = N-1
    HMM.X0, x0 = center(E)    # Decompose ensemble.
    w      = np.zeros(N)  # Control vector for the mean state.
    T      = np.eye(N)    # Anomalies transform matrix.
    Tinv   = np.eye(N)
    # Explicit Tinv [instead of tinv(T)] allows for merging MDA code
    # with iEnKS/EnRML code, and flop savings in 'Sqrt' case.

    for iteration in np.arange(nIter):
        # Reconstruct smoothed ensemble.
        E = x0 + (w + EPS*T)@HMM.X0
        # Forecast.
        for kCycle in DAW:
            for k, t, dt in HMM.tseq.cycle(kCycle):  # noqa
                E = HMM.Dyn(E, t-dt, dt)
        # Observe.
        Eo = HMM.Obs(E, t)

        # Undo the bundle scaling of ensemble.
        if EPS != 1.0:
            E  = inflate_ens(E, 1/EPS)
            Eo = inflate_ens(Eo, 1/EPS)

        # Assess forecast stats; store {Xf, T_old} for analysis assessment.
        if iteration == 0:
            stats.assess(k, ko, 'f', E=E)
            Xf, xf = center(E)
        T_old = T

        # Prepare analysis.
        Y, xo  = center(Eo)         # Get HMM.Obs {anomalies, mean}.
        dy     = (y - xo) @ Rm12.T  # Transform HMM.Obs space.
        Y      = Y        @ Rm12.T  # Transform HMM.Obs space.
        Y0     = Tinv @ Y           # "De-condition" the HMM.Obs anomalies.
        V, s, UT = svd0(Y0)         # Decompose Y0.

        # Set "cov normlzt fctr" za ("effective ensemble size")
        # => pre_infl^2 = (N-1)/za.
        if xN is None:
            za  = N1
        else:
            za  = zeta_a(*hyperprior_coeffs(s, N, xN), w)
        if MDA:
            # inflation (factor: nIter) of the ObsErrCov.
            za *= nIter

        # Post. cov (approx) of w,
        # estimated at current iteration, raised to power.
        def Cowp(expo): return (V * (pad0(s**2, N) + za)**-expo) @ V.T
        Cow1 = Cowp(1.0)

        if MDA:  # View update as annealing (progressive assimilation).
            Cow1 = Cow1 @ T  # apply previous update
            dw = dy @ Y.T @ Cow1
            if 'PertObs' in upd_a:   # == "ES-MDA". By Emerick/Reynolds
                D   = mean0(np.random.randn(*Y.shape)) * np.sqrt(nIter)
                T  -= (Y + D) @ Y.T @ Cow1
            elif 'Sqrt' in upd_a:    # == "ETKF-ish". By Raanes
                T   = Cowp(0.5) * np.sqrt(za) @ T
            elif 'Order1' in upd_a:  # == "DEnKF-ish". By Emerick
                T  -= 0.5 * Y @ Y.T @ Cow1
            # Tinv = eye(N) [as initialized] coz MDA does not de-condition.

        else:  # View update as Gauss-Newton optimzt. of log-posterior.
            grad  = Y0@dy - w*za                  # Cost function gradient
            dw    = grad@Cow1                     # Gauss-Newton step
            # ETKF-ish". By Bocquet/Sakov.
            if 'Sqrt' in upd_a:
                # Sqrt-transforms
                T     = Cowp(0.5) * np.sqrt(N1)
                Tinv  = Cowp(-.5) / np.sqrt(N1)
                # Tinv saves time [vs tinv(T)] when Nx<N
            # "EnRML". By Oliver/Chen/Raanes/Evensen/Stordal.
            elif 'PertObs' in upd_a:
                D     = mean0(np.random.randn(*Y.shape)) \
                    if iteration == 0 else D
                gradT = -(Y+D)@Y0.T + N1*(np.eye(N) - T)
                T     = T + gradT@Cow1
                # Tinv= tinv(T, threshold=N1)  # unstable
                Tinv  = sla.inv(T+1)           # the +1 is for stability.
            # "DEnKF-ish". By Raanes.
            elif 'Order1' in upd_a:
                # Included for completeness; does not make much sense.
                gradT = -0.5*Y@Y0.T + N1*(np.eye(N) - T)
                T     = T + gradT@Cow1
                Tinv  = tinv(T, threshold=N1)

        w += dw
        if dw@dw < wtol*N:
            break

    # Assess (analysis) stats.
    # The final_increment is a linearization to
    # (i) avoid re-running the model and
    # (ii) reproduce EnKF in case nIter==1.
    final_increment = (dw+T-T_old)@Xf
    # See docs/snippets/iEnKS_Ea.jpg.
    stats.assess(k, ko, 'a', E=E+final_increment)
    stats.iters[ko] = iteration+1
    if xN:
        stats.infl[ko] = np.sqrt(N1/za)

    # Final (smoothed) estimate of E at [kLag].
    E = x0 + (w+T)@HMM.X0

    return E


@var_method
class Var4D:
    """4D-Var.

    Cycling scheme is same as in iEnKS (i.e. the shift is always 1*ko).

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
        R, Ko = HMM.Obs.noise.C, HMM.tseq.Ko
        Rm12 = R.sym_sqrt_inv
        Nx = HMM.Dyn.M

        # Set background covariance. Note that it is static (compare to iEnKS).
        if isinstance(self.B, np.ndarray):
            # compare ndarray 1st to avoid == error for ndarray
            B = self.B.astype(float)
        elif self.B in (None, 'clim'):
            # Use climatological cov, estimated from truth
            B = np.cov(xx.T)
        elif self.B == 'eye':
            B = np.eye(Nx)
        else:
            raise ValueError("Bad input B.")
        B *= self.xB
        B12 = CovMat(B).sym_sqrt

        # Init
        x = HMM.X0.mu
        self.stats.assess(0, mu=x, Cov=B)

        # Loop over DA windows (DAW).
        for ko in progbar(np.arange(-1, Ko+self.Lag+1)):
            kLag = ko-self.Lag
            DAW = range(max(0, kLag+1), min(ko, Ko) + 1)

            # Assimilation (if ∃ "not-fully-assimlated" Obs).
            if 0 <= ko <= Ko:

                # Init iterations.
                w   = np.zeros(Nx)  # Control vector for the mean state.
                x0  = x.copy()      # Increment reference.

                for iteration in np.arange(self.nIter):
                    # Reconstruct smoothed state.
                    x = x0 + B12@w
                    X = B12  # Aggregate composite TLMs onto B12
                    # Forecast.
                    for kCycle in DAW:
                        for k, t, dt in HMM.tseq.cycle(kCycle):  # noqa
                            X = HMM.Dyn.linear(x, t-dt, dt) @ X
                            x = HMM.Dyn(x, t-dt, dt)

                    # Assess forecast self.stats
                    if iteration == 0:
                        self.stats.assess(k, ko, 'f', mu=x, Cov=X@X.T)

                    # Observe.
                    Y  = HMM.Obs.linear(x, t) @ X
                    xo = HMM.Obs(x, t)

                    # Analysis prep.
                    y      = yy[ko]          # Get current HMM.Obs.
                    dy     = Rm12 @ (y - xo)   # Transform HMM.Obs space.
                    Y      = Rm12 @ Y          # Transform HMM.Obs space.
                    V, s, UT = svd0(Y.T)       # Decomp for lin-alg update comps.

                    # Post. cov (approx) of w,
                    # estimated at current iteration, raised to power.
                    Cow1 = (V * (pad0(s**2, Nx) + 1)**-1.0) @ V.T

                    # Compute analysis update.
                    grad = Y.T@dy - w          # Cost function gradient
                    dw   = Cow1@grad           # Gauss-Newton step
                    w   += dw                  # Step

                    if dw@dw < self.wtol*Nx:
                        break

                # Assess (analysis) self.stats.
                final_increment = X@dw
                self.stats.assess(k, ko, 'a', mu=x+final_increment, Cov=X@Cow1@X.T)
                self.stats.iters[ko] = iteration+1

                # Final (smoothed) estimate at [kLag].
                x = x0 + B12@w
                X = B12

            # Slide/shift DAW by propagating smoothed ('s') state from [kLag].
            if -1 <= kLag < Ko:
                if kLag >= 0:
                    self.stats.assess(HMM.tseq.kko[kLag], kLag, 's',
                                      mu=x, Cov=X@Cow1@X.T)
                for k, t, dt in HMM.tseq.cycle(kLag+1):
                    self.stats.assess(k-1, None, 'u', mu=x, Cov=Y@Y.T)
                    X = HMM.Dyn.linear(x, t-dt, dt) @ X
                    x = HMM.Dyn(x, t-dt, dt)

        self.stats.assess(k, Ko, 'us', mu=x, Cov=X@Cow1@X.T)
