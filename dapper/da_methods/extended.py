"""The extended KF (EKF) and the (Rauch-Tung-Striebel) smoother."""

import numpy as np

from dapper.tools.linalg import mrdiv
from dapper.tools.progressbar import progbar

from . import da_method


@da_method()
class ExtKF:
    """The extended Kalman filter.

    If everything is linear-Gaussian, this provides the exact solution
    to the Bayesian filtering equations.

    - infl (inflation) may be specified.
      Default: 1.0 (i.e. none), as is optimal in the lin-Gauss case.
      Gets applied at each dt, with infl_per_dt := inlf**(dt), so that
      infl_per_unit_time == infl.
      Specifying it this way (per unit time) means less tuning.
    """

    infl: float = 1.0

    def assimilate(self, HMM, xx, yy):
        R  = HMM.Obs.noise.C.full
        Q  = 0 if HMM.Dyn.noise.C == 0 else HMM.Dyn.noise.C.full

        mu = HMM.X0.mu
        P  = HMM.X0.C.full

        self.stats.assess(0, mu=mu, Cov=P)

        for k, ko, t, dt in progbar(HMM.tseq.ticker):

            mu = HMM.Dyn(mu, t-dt, dt)
            F  = HMM.Dyn.linear(mu, t-dt, dt)
            P  = self.infl**(dt)*(F@P@F.T) + dt*Q

            # Of academic interest? Higher-order linearization:
            # mu_i += 0.5 * (Hessian[f_i] * P).sum()

            if ko is not None:
                self.stats.assess(k, ko, 'f', mu=mu, Cov=P)
                H  = HMM.Obs.linear(mu, t)
                KG = mrdiv(P @ H.T, H@P@H.T + R)
                y  = yy[ko]
                mu = mu + KG@(y - HMM.Obs(mu, t))
                KH = KG@H
                P  = (np.eye(HMM.Dyn.M) - KH) @ P

                self.stats.trHK[ko] = KH.trace()/HMM.Dyn.M

            self.stats.assess(k, ko, mu=mu, Cov=P)


# TODO 5: Clean up
@da_method()
class ExtRTS:
    """The extended Rauch-Tung-Striebel (or "two-pass") smoother."""

    infl: float = 1.0
    DeCorr: float = 1.0

    def assimilate(self, HMM, xx, yy):
        Nx = HMM.Dyn.M

        R  = HMM.Obs.noise.C.full
        Q  = 0 if HMM.Dyn.noise.C == 0 else HMM.Dyn.noise.C.full

        mu    = np.zeros((HMM.tseq.K+1, Nx))
        P     = np.zeros((HMM.tseq.K+1, Nx, Nx))

        # Forecasted values
        muf   = np.zeros((HMM.tseq.K+1, Nx))
        Pf    = np.zeros((HMM.tseq.K+1, Nx, Nx))
        Ff    = np.zeros((HMM.tseq.K+1, Nx, Nx))

        mu[0] = HMM.X0.mu
        P[0] = HMM.X0.C.full

        self.stats.assess(0, mu=mu[0], Cov=P[0])

        # Forward pass
        for k, ko, t, dt in progbar(HMM.tseq.ticker, 'ExtRTS->'):
            mu[k]  = HMM.Dyn(mu[k-1], t-dt, dt)
            F      = HMM.Dyn.linear(mu[k-1], t-dt, dt)
            P[k]   = self.infl**(dt)*(F@P[k-1]@F.T) + dt*Q

            # Store forecast and Jacobian
            muf[k] = mu[k]
            Pf[k]  = P[k]
            Ff[k]  = F

            if ko is not None:
                self.stats.assess(k, ko, 'f', mu=mu[k], Cov=P[k])
                H     = HMM.Obs.linear(mu[k], t)
                KG    = mrdiv(P[k] @ H.T, H@P[k]@H.T + R)
                y     = yy[ko]
                mu[k] = mu[k] + KG@(y - HMM.Obs(mu[k], t))
                KH    = KG@H
                P[k]  = (np.eye(Nx) - KH) @ P[k]
                self.stats.assess(k, ko, 'a', mu=mu[k], Cov=P[k])

        # Backward pass
        for k in progbar(range(HMM.tseq.K)[::-1], 'ExtRTS<-'):
            J     = mrdiv(P[k]@Ff[k+1].T, Pf[k+1])
            J    *= self.DeCorr
            mu[k] = mu[k]  + J @ (mu[k+1]  - muf[k+1])
            P[k]  = P[k] + J @ (P[k+1] - Pf[k+1]) @ J.T
        for k in progbar(range(HMM.tseq.K+1), desc='Assess'):
            self.stats.assess(k, mu=mu[k], Cov=P[k])
