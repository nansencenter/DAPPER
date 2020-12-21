""""Unsophisticated" but robust (widely applicable) DA methods.

Many are based on `bib.raanes2016thesis`.
"""
from typing import Optional

import numpy as np

import dapper.tools.series as series
from dapper.stats import center
from dapper.tools.linalg import mrdiv
from dapper.tools.matrices import CovMat
from dapper.tools.progressbar import progbar

from . import da_method


@da_method()
class Climatology:
    """A baseline/reference method.

    Note that the "climatology" is computed from truth, which might be
    (unfairly) advantageous if the simulation is too short (vs mixing time)."""

    def assimilate(self, HMM, xx, yy):
        chrono, stats = HMM.t, self.stats

        muC = np.mean(xx, 0)
        AC  = xx - muC
        PC  = CovMat(AC, 'A')

        stats.assess(0, mu=muC, Cov=PC)
        stats.trHK[:] = 0

        for k, kObs, _, _ in progbar(chrono.ticker):
            fau = 'u' if kObs is None else 'fau'
            stats.assess(k, kObs, fau, mu=muC, Cov=PC)


@da_method()
class OptInterp:
    """Optimal Interpolation -- a baseline/reference method.

    Uses the Kalman filter equations,
    but with a prior from the Climatology."""

    def assimilate(self, HMM, xx, yy):
        Dyn, Obs, chrono, stats = HMM.Dyn, HMM.Obs, HMM.t, self.stats

        # Compute "climatological" Kalman gain
        muC = np.mean(xx, 0)
        AC  = xx - muC
        PC  = (AC.T @ AC) / (xx.shape[0] - 1)

        # Setup scalar "time-series" covariance dynamics.
        # ONLY USED FOR DIAGNOSTICS, not to affect the Kalman gain.
        L  = series.estimate_corr_length(AC.ravel(order='F'))
        SM = fit_sigmoid(1/2, L, 0)

        # Init
        mu = muC
        stats.assess(0, mu=mu, Cov=PC)

        for k, kObs, t, dt in progbar(chrono.ticker):
            # Forecast
            mu = Dyn(mu, t-dt, dt)
            if kObs is not None:
                stats.assess(k, kObs, 'f', mu=muC, Cov=PC)

                # Analysis
                H  = Obs.linear(muC, t)
                KG  = mrdiv(PC@H.T, H@PC@H.T + Obs.noise.C.full)
                mu = muC + KG@(yy[kObs] - Obs(muC, t))

                P  = (np.eye(Dyn.M) - KG@H) @ PC
                SM = fit_sigmoid(P.trace()/PC.trace(), L, k)

            stats.assess(k, kObs, mu=mu, Cov=2*PC*SM(k))


@da_method()
class Var3D:
    """
    3D-Var -- a baseline/reference method.

    This implementation is not "Var"-ish: there is no *iterative* optimzt.
    Instead, it does the full analysis update in one step: the Kalman filter,
    with the background covariance being user specified, through B and xB."""
    B: Optional[np.ndarray] = None
    xB: float               = 1.0

    def assimilate(self, HMM, xx, yy):
        Dyn, Obs, chrono, X0, stats = HMM.Dyn, HMM.Obs, HMM.t, HMM.X0, self.stats

        if self.B in (None, 'clim'):
            # Use climatological cov, ...
            B = np.cov(xx.T)  # ... estimated from truth
        elif self.B == 'eye':
            B = np.eye(HMM.Nx)
        else:
            B = self.B
        B *= self.xB

        # ONLY USED FOR DIAGNOSTICS, not to change the Kalman gain.
        CC = 2*np.cov(xx.T)
        L  = series.estimate_corr_length(center(xx)[0].ravel(order='F'))
        P  = X0.C.full
        SM = fit_sigmoid(P.trace()/CC.trace(), L, 0)

        # Init
        mu = X0.mu
        stats.assess(0, mu=mu, Cov=P)

        for k, kObs, t, dt in progbar(chrono.ticker):
            # Forecast
            mu = Dyn(mu, t-dt, dt)
            P  = CC*SM(k)

            if kObs is not None:
                stats.assess(k, kObs, 'f', mu=mu, Cov=P)

                # Analysis
                H  = Obs.linear(mu, t)
                KG = mrdiv(B@H.T, H@B@H.T + Obs.noise.C.full)
                mu = mu + KG@(yy[kObs] - Obs(mu, t))

                # Re-calibrate fit_sigmoid with new W0 = Pa/B
                P = (np.eye(Dyn.M) - KG@H) @ B
                SM = fit_sigmoid(P.trace()/CC.trace(), L, k)

            stats.assess(k, kObs, mu=mu, Cov=P)


def fit_sigmoid(Sb, L, kb):
    """Return a sigmoid [function S(k)] for approximating error dynamics.

    We use the logistic function for the sigmoid; it's the solution of the
    "population growth" ODE: dS/dt = a*S*(1-S/S(∞)).
    NB: It might be better to use the "error growth ODE" of Lorenz/Dalcher/Kalnay,
    but this has a significantly more complicated closed-form solution,
    and reduces to the above ODE when there's no model error (ODE source term).

    The "normalized" sigmoid, S1, is symmetric around 0, and S1(-∞)=0 and S1(∞)=1.

    The sigmoid S(k) = S1(a*(k-kb) + b) is fitted (see docs/snippets/sigmoid.jpg) with

    - a corresponding to a given corr. length L.
    - b to match values of S(kb) and Sb"""

    def sigmoid(k): return 1/(1+np.exp(-k))  # normalized sigmoid
    def inv_sig(s): return np.log(s/(1-s))  # its inverse

    a = 1/L
    b = inv_sig(Sb)

    def S(k):
        return sigmoid(b + a*(k-kb))

    return S


@da_method()
class EnCheat:
    """A baseline/reference method.

    Should be implemented as part of Stats instead."""

    def assimilate(self, HMM, xx, yy): pass
