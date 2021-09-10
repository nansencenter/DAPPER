"""More experimental or esoteric DA methods."""

import numpy as np

from dapper.da_methods.ensemble import add_noise, post_process, serial_inds
from dapper.da_methods.particle import reweight
from dapper.stats import center
from dapper.tools.matrices import funm_psd
from dapper.tools.progressbar import progbar

from .ensemble import ens_method


@ens_method
class RHF:
    """Rank histogram filter.

    Refs: `bib.anderson2010non`.

    Quick & dirty implementation without attention to (de)tails.
    """

    N: int
    ordr: str = 'rand'

    def assimilate(self, HMM, xx, yy):
        Dyn, Obs, tseq, X0, stats, N = \
            HMM.Dyn, HMM.Obs, HMM.tseq, HMM.X0, self.stats, self.N

        N1       = N-1
        step     = 1/N
        cdf_grid = np.linspace(step/2, 1-step/2, N)

        R    = Obs.noise
        Rm12 = Obs.noise.C.sym_sqrt_inv

        E = X0.sample(N)
        stats.assess(0, E=E)

        for k, ko, t, dt in progbar(tseq.ticker):
            E = Dyn(E, t-dt, dt)
            E = add_noise(E, dt, Dyn.noise, self.fnoise_treatm)

            if ko is not None:
                stats.assess(k, ko, 'f', E=E)
                y    = yy[ko]
                inds = serial_inds(self.ordr, y, R, center(E)[0])

                for _, j in enumerate(inds):
                    Eo = Obs(E, t)
                    xo = np.mean(Eo, 0)
                    Y  = Eo - xo
                    mu = np.mean(E, 0)
                    A  = E-mu

                    # Update j-th component of observed ensemble
                    dYf    = Rm12[j, :] @ (y - Eo).T  # NB: does Rm12 make sense?
                    Yj     = Rm12[j, :] @ Y.T
                    Regr   = A.T@Yj/np.sum(Yj**2)

                    Sorted = np.argsort(dYf)
                    Revert = np.argsort(Sorted)
                    dYf    = dYf[Sorted]
                    w      = reweight(np.ones(N), innovs=dYf[:, None])  # Lklhd
                    w      = w.clip(1e-10)  # Avoid zeros in interp1
                    cw     = w.cumsum()
                    cw    /= cw[-1]
                    cw    *= N1/N
                    cdfs   = np.minimum(np.maximum(cw[0], cdf_grid), cw[-1])
                    dhE    = -dYf + np.interp(cdfs, cw, dYf)
                    dhE    = dhE[Revert]
                    # Update state by regression
                    E     += np.outer(-dhE, Regr)

                E = post_process(E, self.infl, self.rot)

            stats.assess(k, ko, E=E)


@ens_method
class LNETF:
    """The Nonlinear-Ensemble-Transform-Filter (localized).

    Refs: `bib.wiljes2016second`, `bib.todter2015second`.

    It is (supposedly) a deterministic upgrade of the NLEAF of `bib.lei2011moment`.
    """

    N: int
    loc_rad: float
    taper: str = 'GC'
    Rs: float  = 1.0

    def assimilate(self, HMM, xx, yy):
        Dyn, Obs, tseq, X0, stats = HMM.Dyn, HMM.Obs, HMM.tseq, HMM.X0, self.stats
        Rm12 = Obs.noise.C.sym_sqrt_inv

        E = X0.sample(self.N)
        stats.assess(0, E=E)

        for k, ko, t, dt in progbar(tseq.ticker):
            E = Dyn(E, t-dt, dt)
            E = add_noise(E, dt, Dyn.noise, self.fnoise_treatm)

            if ko is not None:
                stats.assess(k, ko, 'f', E=E)
                mu = np.mean(E, 0)
                A  = E - mu

                Eo = Obs(E, t)
                xo = np.mean(Eo, 0)
                YR = (Eo-xo)  @ Rm12.T
                yR = (yy[ko] - xo) @ Rm12.T

                state_batches, obs_taperer = Obs.localizer(
                    self.loc_rad, 'x2y', t, self.taper)
                for ii in state_batches:
                    # Localize obs
                    jj, tapering = obs_taperer(ii)
                    if len(jj) == 0:
                        return

                    Y_jj  = YR[:, jj] * np.sqrt(tapering)
                    dy_jj = yR[jj]   * np.sqrt(tapering)

                    # NETF:
                    # This "paragraph" is the only difference to the LETKF.
                    innovs = (dy_jj-Y_jj)/self.Rs
                    if 'laplace' in str(type(Obs.noise)).lower():
                        w    = laplace_lklhd(innovs)
                    else:  # assume Gaussian
                        w    = reweight(np.ones(self.N), innovs=innovs)
                    dmu    = w@A[:, ii]
                    AT     = np.sqrt(self.N)*funm_psd(np.diag(w) -
                                                      np.outer(w, w), np.sqrt)@A[:, ii]

                    E[:, ii] = mu[ii] + dmu + AT

                E = post_process(E, self.infl, self.rot)
            stats.assess(k, ko, E=E)


def laplace_lklhd(xx):
    """Compute a Laplacian likelihood.

    Compute likelihood of xx wrt. the sampling distribution
    RVs.LaplaceParallelRV(C=I), i.e., for x in xx:
    p(x) = exp(-sqrt(2)*|x|_1) / sqrt(2).
    """
    logw   = -np.sqrt(2)*np.sum(np.abs(xx), axis=1)
    logw  -= logw.max()      # Avoid numerical error
    w      = np.np.exp(logw)  # non-log
    w     /= w.sum()         # normalize
    return w
