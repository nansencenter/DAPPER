"""The EnKF and other ensemble-based methods."""

import numpy as np
import numpy.random as rnd
import scipy.linalg as sla
from numpy import diag, eye, sqrt, zeros

import dapper.tools.multiproc as mp
from dapper.stats import center, mean0
from dapper.tools.linalg import mldiv, mrdiv, pad0, svd0, svdi, tinv, tsvd
from dapper.tools.matrices import funm_psd, genOG_1
from dapper.tools.progressbar import progbar

from . import da_method


@da_method
class ens_method:
    """Declare default ensemble arguments."""

    infl: float        = 1.0
    rot: bool          = False
    fnoise_treatm: str = 'Stoch'


@ens_method
class EnKF:
    """The ensemble Kalman filter.

    Refs: `bib.evensen2009ensemble`.
    """

    upd_a: str
    N: int

    def assimilate(self, HMM, xx, yy):
        # Init
        E = HMM.X0.sample(self.N)
        self.stats.assess(0, E=E)

        # Cycle
        for k, ko, t, dt in progbar(HMM.tseq.ticker):
            E = HMM.Dyn(E, t-dt, dt)
            E = add_noise(E, dt, HMM.Dyn.noise, self.fnoise_treatm)

            # Analysis update
            if ko is not None:
                self.stats.assess(k, ko, 'f', E=E)
                E = EnKF_analysis(E, HMM.Obs(E, t), HMM.Obs.noise, yy[ko],
                                  self.upd_a, self.stats, ko)
                E = post_process(E, self.infl, self.rot)

            self.stats.assess(k, ko, E=E)


def EnKF_analysis(E, Eo, hnoise, y, upd_a, stats=None, ko=None):
    """Perform the EnKF analysis update.

    This implementation includes several flavours and forms,
    specified by `upd_a`.

    Main references: `bib.sakov2008deterministic`,
    `bib.sakov2008implications`, `bib.hoteit2015mitigating`
    """
    R     = hnoise.C     # Obs noise cov
    N, Nx = E.shape      # Dimensionality
    N1    = N-1          # Ens size - 1

    mu = np.mean(E, 0)   # Ens mean
    A  = E - mu          # Ens anomalies

    xo = np.mean(Eo, 0)  # Obs ens mean
    Y  = Eo-xo           # Obs ens anomalies
    dy = y - xo          # Mean "innovation"

    if 'PertObs' in upd_a:
        # Uses classic, perturbed observations (Burgers'98)
        C  = Y.T @ Y + R.full*N1
        D  = mean0(hnoise.sample(N))
        YC = mrdiv(Y, C)
        KG = A.T @ YC
        HK = Y.T @ YC
        dE = (KG @ (y - D - Eo).T).T
        E  = E + dE

    elif 'Sqrt' in upd_a:
        # Uses a symmetric square root (ETKF)
        # to deterministically transform the ensemble.

        # The various versions below differ only numerically.
        # EVD is default, but for large N use SVD version.
        if upd_a == 'Sqrt' and N > Nx:
            upd_a = 'Sqrt svd'

        if 'explicit' in upd_a:
            # Not recommended due to numerical costs and instability.
            # Implementation using inv (in ens space)
            Pw = sla.inv(Y @ R.inv @ Y.T + N1*eye(N))
            T  = sla.sqrtm(Pw) * sqrt(N1)
            HK = R.inv @ Y.T @ Pw @ Y
            # KG = R.inv @ Y.T @ Pw @ A
        elif 'svd' in upd_a:
            # Implementation using svd of Y R^{-1/2}.
            V, s, _ = svd0(Y @ R.sym_sqrt_inv.T)
            d       = pad0(s**2, N) + N1
            Pw      = (V * d**(-1.0)) @ V.T
            T       = (V * d**(-0.5)) @ V.T * sqrt(N1)
            # docs/snippets/trHK.jpg
            trHK    = np.sum((s**2+N1)**(-1.0) * s**2)
        elif 'sS' in upd_a:
            # Same as 'svd', but with slightly different notation
            # (sometimes used by Sakov) using the normalization sqrt(N1).
            S       = Y @ R.sym_sqrt_inv.T / sqrt(N1)
            V, s, _ = svd0(S)
            d       = pad0(s**2, N) + 1
            Pw      = (V * d**(-1.0))@V.T / N1  # = G/(N1)
            T       = (V * d**(-0.5))@V.T
            # docs/snippets/trHK.jpg
            trHK    = np.sum((s**2 + 1)**(-1.0)*s**2)
        else:  # 'eig' in upd_a:
            # Implementation using eig. val. decomp.
            d, V   = sla.eigh(Y @ R.inv @ Y.T + N1*eye(N))
            T      = V@diag(d**(-0.5))@V.T * sqrt(N1)
            Pw     = V@diag(d**(-1.0))@V.T
            HK     = R.inv @ Y.T @ (V @ diag(d**(-1)) @ V.T) @ Y
        w = dy @ R.inv @ Y.T @ Pw
        E = mu + w@A + T@A

    elif 'Serial' in upd_a:
        # Observations assimilated one-at-a-time:
        inds = serial_inds(upd_a, y, R, A)
        #  Requires de-correlation:
        dy   = dy @ R.sym_sqrt_inv.T
        Y    = Y  @ R.sym_sqrt_inv.T
        # Enhancement in the nonlinear case:
        # re-compute Y each scalar obs assim.
        # But: little benefit, model costly (?),
        # updates cannot be accumulated on S and T.

        if any(x in upd_a for x in ['Stoch', 'ESOPS', 'Var1']):
            # More details: Misc/Serial_ESOPS.py.
            for i, j in enumerate(inds):

                # Perturbation creation
                if 'ESOPS' in upd_a:
                    # "2nd-O exact perturbation sampling"
                    if i == 0:
                        # Init -- increase nullspace by 1
                        V, s, UT = svd0(A)
                        s[N-2:] = 0
                        A = svdi(V, s, UT)
                        v = V[:, N-2]
                    else:
                        # Orthogonalize v wrt. the new A
                        #
                        # v = Zj - Yj (from paper) requires Y==HX.
                        # Instead: mult` should be c*ones(Nx) so we can
                        # project v into ker(A) such that v@A is null.
                        mult  = (v@A) / (Yj@A) # noqa
                        v     = v - mult[0]*Yj # noqa
                        v    /= sqrt(v@v)
                    Zj  = v*sqrt(N1)  # Standardized perturbation along v
                    Zj *= np.sign(rnd.rand()-0.5)  # Random sign
                else:
                    # The usual stochastic perturbations.
                    Zj = mean0(rnd.randn(N))  # Un-coloured noise
                    if 'Var1' in upd_a:
                        Zj *= sqrt(N/(Zj@Zj))

                # Select j-th obs
                Yj  = Y[:, j]       # [j] obs anomalies
                dyj = dy[j]         # [j] innov mean
                DYj = Zj - Yj       # [j] innov anomalies
                DYj = DYj[:, None]  # Make 2d vertical

                # Kalman gain computation
                C     = Yj@Yj + N1  # Total obs cov
                KGx   = Yj @ A / C  # KG to update state
                KGy   = Yj @ Y / C  # KG to update obs

                # Updates
                A    += DYj * KGx
                mu   += dyj * KGx
                Y    += DYj * KGy
                dy   -= dyj * KGy
            E = mu + A
        else:
            # "Potter scheme", "EnSRF"
            # - EAKF's two-stage "update-regress" form yields
            #   the same *ensemble* as this.
            # - The form below may be derived as "serial ETKF",
            #   but does not yield the same
            #   ensemble as 'Sqrt' (which processes obs as a batch)
            #   -- only the same mean/cov.
            T = eye(N)
            for j in inds:
                Yj = Y[:, j]
                C  = Yj@Yj + N1
                Tj = np.outer(Yj, Yj / (C + sqrt(N1*C)))
                T -= Tj @ T
                Y -= Tj @ Y
            w = dy@Y.T@T/N1
            E = mu + w@A + T@A

    elif 'DEnKF' == upd_a:
        # Uses "Deterministic EnKF" (sakov'08)
        C  = Y.T @ Y + R.full*N1
        YC = mrdiv(Y, C)
        KG = A.T @ YC
        HK = Y.T @ YC
        E  = E + KG@dy - 0.5*(KG@Y.T).T

    else:
        raise KeyError("No analysis update method found: '" + upd_a + "'.")

    # Diagnostic: relative influence of observations
    if stats is not None:
        if 'trHK' in locals():
            stats.trHK[ko] = trHK      / hnoise.M
        elif 'HK' in locals():
            stats.trHK[ko] = HK.trace()/hnoise.M

    return E


def post_process(E, infl, rot):
    """Inflate, Rotate.

    To avoid recomputing/recombining anomalies,
    this should have been inside :func:`EnKF_analysis`

    But it is kept as a separate function

    - for readability;
    - to avoid inflating/rotationg smoothed states (for the :func:`EnKS`).
    """
    do_infl = infl != 1.0 and infl != '-N'

    if do_infl or rot:
        A, mu  = center(E)
        N, Nx  = E.shape
        T      = eye(N)

        if do_infl:
            T = infl * T

        if rot:
            T = genOG_1(N, rot) @ T

        E = mu + T@A
    return E


def add_noise(E, dt, noise, method):
    """Treatment of additive noise for ensembles.

    Refs: `bib.raanes2014ext`
    """
    if noise.C == 0:
        return E

    N, Nx = E.shape
    A, mu = center(E)
    Q12   = noise.C.Left
    Q     = noise.C.full

    def sqrt_core():
        T    = np.nan    # cause error if used
        Qa12 = np.nan    # cause error if used
        A2   = A.copy()  # Instead of using (the implicitly nonlocal) A,
        # which changes A outside as well. NB: This is a bug in Datum!
        if N <= Nx:
            Ainv = tinv(A2.T)
            Qa12 = Ainv@Q12
            T    = funm_psd(eye(N) + dt*(N-1)*(Qa12@Qa12.T), sqrt)
            A2   = T@A2
        else:  # "Left-multiplying" form
            P  = A2.T @ A2 / (N-1)
            L  = funm_psd(eye(Nx) + dt*mrdiv(Q, P), sqrt)
            A2 = A2 @ L.T
        E = mu + A2
        return E, T, Qa12

    if method == 'Stoch':
        # In-place addition works (also) for empty [] noise sample.
        E += sqrt(dt)*noise.sample(N)

    elif method == 'none':
        pass

    elif method == 'Mult-1':
        varE   = np.var(E, axis=0, ddof=1).sum()
        ratio  = (varE + dt*diag(Q).sum())/varE
        E      = mu + sqrt(ratio)*A
        E      = svdi(*tsvd(E, 0.999))  # Explained in Datum

    elif method == 'Mult-M':
        varE   = np.var(E, axis=0)
        ratios = sqrt((varE + dt*diag(Q))/varE)
        E      = mu + A*ratios
        E      = svdi(*tsvd(E, 0.999))  # Explained in Datum

    elif method == 'Sqrt-Core':
        E = sqrt_core()[0]

    elif method == 'Sqrt-Mult-1':
        varE0 = np.var(E, axis=0, ddof=1).sum()
        varE2 = (varE0 + dt*diag(Q).sum())
        E, _, Qa12 = sqrt_core()
        if N <= Nx:
            A, mu   = center(E)
            varE1   = np.var(E, axis=0, ddof=1).sum()
            ratio   = varE2/varE1
            E       = mu + sqrt(ratio)*A
            E       = svdi(*tsvd(E, 0.999))  # Explained in Datum

    elif method == 'Sqrt-Add-Z':
        E, _, Qa12 = sqrt_core()
        if N <= Nx:
            Z  = Q12 - A.T@Qa12
            E += sqrt(dt)*(Z@rnd.randn(Z.shape[1], N)).T

    elif method == 'Sqrt-Dep':
        E, T, Qa12 = sqrt_core()
        if N <= Nx:
            # Q_hat12: reuse svd for both inversion and projection.
            Q_hat12      = A.T @ Qa12
            U, s, VT     = tsvd(Q_hat12, 0.99)
            Q_hat12_inv  = (VT.T * s**(-1.0)) @ U.T
            Q_hat12_proj = VT.T@VT
            rQ = Q12.shape[1]
            # Calc D_til
            Z      = Q12 - Q_hat12
            D_hat  = A.T@(T-eye(N))
            Xi_hat = Q_hat12_inv @ D_hat
            Xi_til = (eye(rQ) - Q_hat12_proj)@rnd.randn(rQ, N)
            D_til  = Z@(Xi_hat + sqrt(dt)*Xi_til)
            E     += D_til.T

    else:
        raise KeyError('No such method')

    return E


@ens_method
class EnKS:
    """The ensemble Kalman smoother.

    Refs: `bib.evensen2009ensemble`

    The only difference to the EnKF
    is the management of the lag and the reshapings.
    """

    upd_a: str
    N: int
    Lag: int

    # Reshapings used in smoothers to go to/from
    # 3D arrays, where the 0th axis is the Lag index.
    def reshape_to(self, E):
        K, N, Nx = E.shape
        return E.transpose([1, 0, 2]).reshape((N, K*Nx))

    def reshape_fr(self, E, Nx):
        N, Km = E.shape
        K    = Km//Nx
        return E.reshape((N, K, Nx)).transpose([1, 0, 2])

    def assimilate(self, HMM, xx, yy):
        # Inefficient version, storing full time series ensemble.
        # See iEnKS for a "rolling" version.
        E    = zeros((HMM.tseq.K+1, self.N, HMM.Dyn.M))
        E[0] = HMM.X0.sample(self.N)

        for k, ko, t, dt in progbar(HMM.tseq.ticker):
            E[k] = HMM.Dyn(E[k-1], t-dt, dt)
            E[k] = add_noise(E[k], dt, HMM.Dyn.noise, self.fnoise_treatm)

            if ko is not None:
                self.stats.assess(k, ko, 'f', E=E[k])

                Eo    = HMM.Obs(E[k], t)
                y     = yy[ko]

                # Inds within Lag
                kk    = range(max(0, k-self.Lag*HMM.tseq.dko), k+1)

                EE    = E[kk]

                EE    = self.reshape_to(EE)
                EE    = EnKF_analysis(EE, Eo, HMM.Obs.noise, y,
                                      self.upd_a, self.stats, ko)
                E[kk] = self.reshape_fr(EE, HMM.Dyn.M)
                E[k]  = post_process(E[k], self.infl, self.rot)
                self.stats.assess(k, ko, 'a', E=E[k])

        for k, ko, _, _ in progbar(HMM.tseq.ticker, desc='Assessing'):
            self.stats.assess(k, ko, 'u', E=E[k])
            if ko is not None:
                self.stats.assess(k, ko, 's', E=E[k])


@ens_method
class EnRTS:
    """EnRTS (Rauch-Tung-Striebel) smoother.

    Refs: `bib.raanes2016thesis`
    """

    upd_a: str
    N: int
    DeCorr: float

    def assimilate(self, HMM, xx, yy):
        E    = zeros((HMM.tseq.K+1, self.N, HMM.Dyn.M))
        Ef   = E.copy()
        E[0] = HMM.X0.sample(self.N)

        # Forward pass
        for k, ko, t, dt in progbar(HMM.tseq.ticker):
            E[k]  = HMM.Dyn(E[k-1], t-dt, dt)
            E[k]  = add_noise(E[k], dt, HMM.Dyn.noise, self.fnoise_treatm)
            Ef[k] = E[k]

            if ko is not None:
                self.stats.assess(k, ko, 'f', E=E[k])
                Eo   = HMM.Obs(E[k], t)
                y    = yy[ko]
                E[k] = EnKF_analysis(E[k], Eo, HMM.Obs.noise, y,
                                     self.upd_a, self.stats, ko)
                E[k] = post_process(E[k], self.infl, self.rot)
                self.stats.assess(k, ko, 'a', E=E[k])

        # Backward pass
        for k in progbar(range(HMM.tseq.K)[::-1]):
            A  = center(E[k])[0]
            Af = center(Ef[k+1])[0]

            J = tinv(Af) @ A
            J *= self.DeCorr

            E[k] += (E[k+1] - Ef[k+1]) @ J

        for k, ko, _, _ in progbar(HMM.tseq.ticker, desc='Assessing'):
            self.stats.assess(k, ko, 'u', E=E[k])
            if ko is not None:
                self.stats.assess(k, ko, 's', E=E[k])


def serial_inds(upd_a, y, cvR, A):
    """Get the indices used for serial updating.

    - Default: random ordering
    - if "mono" in `upd_a`: `1, 2, ..., len(y)`
    - if "sorted" in `upd_a`: sort by variance
    """
    if 'mono' in upd_a:
        # Not robust?
        inds = np.arange(len(y))
    elif 'sorted' in upd_a:
        N = len(A)
        dC = cvR.diag
        if np.all(dC == dC[0]):
            # Sort y by P
            dC = np.sum(A*A, 0)/(N-1)
        inds = np.argsort(dC)
    else:  # Default: random ordering
        inds = rnd.permutation(len(y))
    return inds


@ens_method
class SL_EAKF:
    """Serial, covariance-localized EAKF.

    Refs: `bib.karspeck2007experimental`.

    Used without localization, this should be equivalent (full ensemble equality)
    to the `EnKF` with `upd_a='Serial'`.
    """

    N: int
    loc_rad: float
    taper: str  = 'GC'
    ordr: str   = 'rand'

    def assimilate(self, HMM, xx, yy):
        N1   = self.N-1
        R    = HMM.Obs.noise
        Rm12 = HMM.Obs.noise.C.sym_sqrt_inv

        E = HMM.X0.sample(self.N)
        self.stats.assess(0, E=E)

        for k, ko, t, dt in progbar(HMM.tseq.ticker):
            E = HMM.Dyn(E, t-dt, dt)
            E = add_noise(E, dt, HMM.Dyn.noise, self.fnoise_treatm)

            if ko is not None:
                self.stats.assess(k, ko, 'f', E=E)
                y    = yy[ko]
                inds = serial_inds(self.ordr, y, R, center(E)[0])

                state_taperer = HMM.Obs.localizer(self.loc_rad, 'y2x', t, self.taper)
                for j in inds:
                    # Prep:
                    # ------------------------------------------------------
                    Eo = HMM.Obs(E, t)
                    xo = np.mean(Eo, 0)
                    Y  = Eo - xo
                    mu = np.mean(E, 0)
                    A  = E-mu
                    # Update j-th component of observed ensemble:
                    # ------------------------------------------------------
                    Y_j    = Rm12[j, :] @ Y.T
                    dy_j   = Rm12[j, :] @ (y - xo)
                    # Prior var * N1:
                    sig2_j = Y_j@Y_j
                    if sig2_j < 1e-9:
                        continue
                    # Update (below, we drop the locality subscript: _j)
                    sig2_u = 1/(1/sig2_j + 1/N1)      # Postr. var * N1
                    alpha  = (N1/(N1+sig2_j))**(0.5)  # Update contraction factor
                    dy2    = sig2_u * dy_j/N1         # Mean update
                    Y2     = alpha*Y_j                # Anomaly update
                    # Update state (regress update from obs space, using localization)
                    # ------------------------------------------------------
                    ii, tapering = state_taperer(j)
                    # ii, tapering = ..., 1  # cancel localization
                    if len(ii) == 0:
                        continue
                    Xi = A[:, ii]*tapering
                    Regression = Xi.T @ Y_j/np.sum(Y_j**2)
                    mu[ii] += Regression*dy2
                    A[:, ii] += np.outer(Y2 - Y_j, Regression)
                    E = mu + A

                E = post_process(E, self.infl, self.rot)

            self.stats.assess(k, ko, E=E)


@ens_method
class LETKF:
    """Same as EnKF (sqrt), but with localization.

    Refs: `bib.hunt2007efficient`.

    NB: Multiproc. yields slow-down for `dapper.mods.Lorenz96`,
    even with `batch_size=(1,)`. But for `dapper.mods.QG`
    (`batch_size=(2,2)` or less) it is quicker.

    NB: If `len(ii)` is small, analysis may be slowed-down with '-N' infl.
    """

    N: int
    loc_rad: float
    taper: str = 'GC'
    xN: float  = 1.0
    g: int     = 0
    mp: bool   = False

    def assimilate(self, HMM, xx, yy):
        N = self.N
        R, N1 = HMM.Obs.noise.C, N-1

        _map = mp.map if self.mp else map

        E = HMM.X0.sample(N)
        self.stats.assess(0, E=E)

        for k, ko, t, dt in progbar(HMM.tseq.ticker):
            # Forecast
            E = HMM.Dyn(E, t-dt, dt)
            E = add_noise(E, dt, HMM.Dyn.noise, self.fnoise_treatm)

            if ko is not None:
                self.stats.assess(k, ko, 'f', E=E)

                # Decompose ensmeble
                mu = np.mean(E, 0)
                A  = E - mu
                # Obs space variables
                y     = yy[ko]
                Y, xo = center(HMM.Obs(E, t))
                # Transform obs space
                Y  = Y        @ R.sym_sqrt_inv.T
                dy = (y - xo) @ R.sym_sqrt_inv.T

                # Local analyses
                # Get localization configuration
                state_batches, obs_taperer = \
                    HMM.Obs.localizer(self.loc_rad, 'x2y', t, self.taper)
                # Avoid pickling self
                xN, g, infl = self.xN, self.g, self.infl

                def local_analysis(ii):
                    """Do the local analysis.

                    Notation:

                    - `ii`: inds. for the state batch defining the locality
                    - `jj`: inds. for the associated obs
                    """
                    # Locate local obs
                    jj, tapering = obs_taperer(ii)
                    if len(jj) == 0:
                        return E[:, ii], N1  # no update
                    Y_jj   = Y[:, jj]
                    dy_jj  = dy[jj]

                    # Adaptive inflation
                    za = effective_N(Y_jj, dy_jj, xN, g) if infl == '-N' else N1

                    # Taper
                    Y_jj  *= sqrt(tapering)
                    dy_jj *= sqrt(tapering)

                    # Compute ETKF update
                    if len(jj) < N:
                        # SVD version
                        V, sd, _ = svd0(Y_jj)
                        d      = pad0(sd**2, N) + za
                        Pw     = (V * d**(-1.0)) @ V.T
                        T      = (V * d**(-0.5)) @ V.T * sqrt(za)
                    else:
                        # EVD version
                        d, V  = sla.eigh(Y_jj@Y_jj.T + za*eye(N))
                        T     = V@diag(d**(-0.5))@V.T * sqrt(za)
                        Pw    = V@diag(d**(-1.0))@V.T
                    AT  = T @ A[:, ii]
                    dmu = dy_jj @ Y_jj.T @ Pw @ A[:, ii]
                    Eii = mu[ii] + dmu + AT
                    return Eii, za

                # Run local analyses
                EE, za = zip(*_map(local_analysis, state_batches))
                for ii, Eii in zip(state_batches, EE):
                    E[:, ii] = Eii

                # Global post-processing
                E = post_process(E, self.infl, self.rot)

                self.stats.infl[ko] = sqrt(N1/np.mean(za))

            self.stats.assess(k, ko, E=E)


def effective_N(YR, dyR, xN, g):
    """Effective ensemble size N.

    As measured by the finite-size EnKF-N
    """
    N, Ny = YR.shape
    N1   = N-1

    V, s, UT = svd0(YR)
    du     = UT @ dyR

    eN, cL = hyperprior_coeffs(s, N, xN, g)

    def pad_rk(arr): return pad0(arr, min(N, Ny))
    def dgn_rk(l1): return pad_rk((l1*s)**2) + N1

    # Make dual cost function (in terms of l1)
    def J(l1):
        val = np.sum(du**2/dgn_rk(l1)) \
            + eN/l1**2 \
            + cL*np.log(l1**2)
        return val

    # Derivatives (not required with minimize_scalar):
    def Jp(l1):
        val = -2*l1   * np.sum(pad_rk(s**2) * du**2/dgn_rk(l1)**2) \
            + -2*eN/l1**3 \
            + 2*cL/l1
        return val

    def Jpp(l1):
        val = 8*l1**2 * np.sum(pad_rk(s**4) * du**2/dgn_rk(l1)**3) \
            + 6*eN/l1**4 \
            + -2*cL/l1**2
        return val

    # Find inflation factor (optimize)
    l1 = Newton_m(Jp, Jpp, 1.0)
    # l1 = fmin_bfgs(J, x0=[1], gtol=1e-4, disp=0)
    # l1 = minimize_scalar(J, bracket=(sqrt(prior_mode), 1e2), tol=1e-4).x

    za = N1/l1**2
    return za


# Notes on optimizers for the 'dual' EnKF-N:
# ----------------------------------------
#  Using minimize_scalar:
#  - Doesn't take dJdx. Advantage: only need J
#  - method='bounded' not necessary and slower than 'brent'.
#  - bracket not necessary either...
#  Using multivariate minimization: fmin_cg, fmin_bfgs, fmin_ncg
#  - these also accept dJdx. But only fmin_bfgs approaches
#    the speed of the scalar minimizers.
#  Using scalar root-finders:
#  - brenth(dJ1, LowB, 1e2,     xtol=1e-6) # Same speed as minimization
#  - newton(dJ1,1.0, fprime=dJ2, tol=1e-6) # No improvement
#  - newton(dJ1,1.0, fprime=dJ2, tol=1e-6, fprime2=dJ3) # No improvement
#  - Newton_m(dJ1,dJ2, 1.0) # Significantly faster. Also slightly better CV?
# => Despite inconvienience of defining analytic derivatives,
#    Newton_m seems like the best option.
#  - In extreme (or just non-linear Obs.mod) cases,
#    the EnKF-N cost function may have multiple minima.
#    Then: should use more robust optimizer!
#
# For 'primal'
# ----------------------------------------
# Similarly, Newton_m seems like the best option,
# although alternatives are provided (commented out).
#
def Newton_m(fun, deriv, x0, is_inverted=False,
             conf=1.0, xtol=1e-4, ytol=1e-7, itermax=10**2):
    """Find root of `fun`.

    This is a simple (and pretty fast) implementation of Newton's method.
    """
    itr = 0
    dx = np.inf
    Jx = fun(x0)

    def norm(x):
        return sqrt(np.sum(x**2))
    while ytol < norm(Jx) and xtol < norm(dx) and itr < itermax:
        Dx  = deriv(x0)
        if is_inverted:
            dx  = Dx @ Jx
        elif isinstance(Dx, float):
            dx  = Jx/Dx
        else:
            dx  = mldiv(Dx, Jx)
        dx *= conf
        x0 -= dx
        Jx  = fun(x0)
        itr += 1
    return x0


def hyperprior_coeffs(s, N, xN=1, g=0):
    r"""Set EnKF-N inflation hyperparams.

    The EnKF-N prior may be specified by the constants:

    - `eN`: Effect of unknown mean
    - `cL`: Coeff in front of log term

    These are trivial constants in the original EnKF-N,
    but are further adjusted (corrected and tuned) for the following reasons.

    - Reason 1: mode correction.
      These parameters bridge the Jeffreys (`xN=1`) and Dirac (`xN=Inf`) hyperpriors
      for the prior covariance, B, as discussed in `bib.bocquet2015expanding`.
      Indeed, mode correction becomes necessary when $$ R \rightarrow \infty $$
      because then there should be no ensemble update (and also no inflation!).
      More specifically, the mode of `l1`'s should be adjusted towards 1
      as a function of $$ I - K H $$ ("prior's weight").
      PS: why do we leave the prior mode below 1 at all?
      Because it sets up "tension" (negative feedback) in the inflation cycle:
      the prior pulls downwards, while the likelihood tends to pull upwards.

    - Reason 2: Boosting the inflation prior's certainty from N to xN*N.
      The aim is to take advantage of the fact that the ensemble may not
      have quite as much sampling error as a fully stochastic sample,
      as illustrated in section 2.1 of `bib.raanes2019adaptive`.

    - Its damping effect is similar to work done by J. Anderson.

    The tuning is controlled by:

    - `xN=1`: is fully agnostic, i.e. assumes the ensemble is generated
      from a highly chaotic or stochastic model.
    - `xN>1`: increases the certainty of the hyper-prior,
      which is appropriate for more linear and deterministic systems.
    - `xN<1`: yields a more (than 'fully') agnostic hyper-prior,
      as if N were smaller than it truly is.
    - `xN<=0` is not meaningful.
    """
    N1 = N-1

    eN = (N+1)/N
    cL = (N+g)/N1

    # Mode correction (almost) as in eqn 36 of `bib.bocquet2015expanding`
    prior_mode = eN/cL                        # Mode of l1 (before correction)
    diagonal   = pad0(s**2, N) + N1           # diag of Y@R.inv@Y + N1*I
    #                                           (Hessian of J)
    I_KH       = np.mean(diagonal**(-1))*N1   # ≈ 1/(1 + HBH/R)
    # I_KH      = 1/(1 + (s**2).sum()/N1)     # Scalar alternative: use tr(HBH/R).
    mc         = sqrt(prior_mode**I_KH)       # Correction coeff

    # Apply correction
    eN /= mc
    cL *= mc

    # Boost by xN
    eN *= xN
    cL *= xN

    return eN, cL


def zeta_a(eN, cL, w):
    """EnKF-N inflation estimation via w.

    Returns `zeta_a = (N-1)/pre-inflation^2`.

    Using this inside an iterative minimization as in the
    `dapper.da_methods.variational.iEnKS` effectively blends
    the distinction between the primal and dual EnKF-N.
    """
    N  = len(w)
    N1 = N-1
    za = N1*cL/(eN + w@w)
    return za


@ens_method
class EnKF_N:
    """Finite-size EnKF (EnKF-N).

    Refs: `bib.bocquet2011ensemble`, `bib.bocquet2015expanding`

    This implementation is pedagogical, prioritizing the "dual" form.
    In consequence, the efficiency of the "primal" form suffers a bit.
    The primal form is included for completeness and to demonstrate equivalence.
    In `dapper.da_methods.variational.iEnKS`, however,
    the primal form is preferred because it
    already does optimization for w (as treatment for nonlinear models).

    `infl` should be unnecessary (assuming no model error, or that Q is correct).

    `Hess`: use non-approx Hessian for ensemble transform matrix?

    `g` is the nullity of A (state anomalies's), ie. g=max(1,N-Nx),
    compensating for the redundancy in the space of w.
    But we have made it an input argument instead, with default 0,
    because mode-finding (of p(x) via the dual) completely ignores this redundancy,
    and the mode gets (undesireably) modified by g.

    `xN` allows tuning the hyper-prior for the inflation.
    Usually, I just try setting it to 1 (default), or 2.
    Further description in hyperprior_coeffs().
    """

    N: int
    dual: bool = False
    Hess: bool = False
    xN: float  = 1.0
    g: int     = 0

    def assimilate(self, HMM, xx, yy):
        R, N, N1 = HMM.Obs.noise.C, self.N, self.N-1

        # Init
        E = HMM.X0.sample(N)
        self.stats.assess(0, E=E)

        # Cycle
        for k, ko, t, dt in progbar(HMM.tseq.ticker):
            # Forecast
            E = HMM.Dyn(E, t-dt, dt)
            E = add_noise(E, dt, HMM.Dyn.noise, self.fnoise_treatm)

            # Analysis
            if ko is not None:
                self.stats.assess(k, ko, 'f', E=E)
                Eo = HMM.Obs(E, t)
                y  = yy[ko]

                mu = np.mean(E, 0)
                A  = E - mu

                xo = np.mean(Eo, 0)
                Y  = Eo-xo
                dy = y - xo

                V, s, UT = svd0(Y @ R.sym_sqrt_inv.T)
                du       = UT @ (dy @ R.sym_sqrt_inv.T)
                def dgn_N(l1): return pad0((l1*s)**2, N) + N1

                # Adjust hyper-prior
                # xN_ = noise_level(self.xN, self.stats, HMM.tseq, N1, ko, A,
                #                   locals().get('A_old', None))
                eN, cL = hyperprior_coeffs(s, N, self.xN, self.g)

                if self.dual:
                    # Make dual cost function (in terms of l1)
                    def pad_rk(arr): return pad0(arr, min(N, HMM.Obs.M))
                    def dgn_rk(l1): return pad_rk((l1*s)**2) + N1

                    def J(l1):
                        val = np.sum(du**2/dgn_rk(l1)) \
                            + eN/l1**2 \
                            + cL*np.log(l1**2)
                        return val

                    # Derivatives (not required with minimize_scalar):
                    def Jp(l1):
                        val = -2*l1 * np.sum(pad_rk(s**2) * du**2/dgn_rk(l1)**2) \
                            + -2*eN/l1**3 + 2*cL/l1
                        return val

                    def Jpp(l1):
                        val = 8*l1**2 * np.sum(pad_rk(s**4) * du**2/dgn_rk(l1)**3) \
                            + 6*eN/l1**4 + -2*cL/l1**2
                        return val
                    # Find inflation factor (optimize)
                    l1 = Newton_m(Jp, Jpp, 1.0)
                    # l1 = fmin_bfgs(J, x0=[1], gtol=1e-4, disp=0)
                    # l1 = minimize_scalar(J, bracket=(sqrt(prior_mode), 1e2),
                    #                      tol=1e-4).x

                else:
                    # Primal form, in a fully linearized version.
                    def za(w): return zeta_a(eN, cL, w)

                    def J(w): return \
                        .5*np.sum(((dy-w@Y)@R.sym_sqrt_inv.T)**2) + \
                        .5*N1*cL*np.log(eN + w@w)
                    # Derivatives (not required with fmin_bfgs):
                    def Jp(w): return -Y@R.inv@(dy-w@Y) + w*za(w)
                    # Jpp   = lambda w:  Y@R.inv@Y.T + \
                    #     za(w)*(eye(N) - 2*np.outer(w,w)/(eN + w@w))
                    # Approx: no radial-angular cross-deriv:
                    # Jpp   = lambda w:  Y@R.inv@Y.T + za(w)*eye(N)

                    def nvrs(w):
                        # inverse of Jpp-approx
                        return (V * (pad0(s**2, N) + za(w)) ** -1.0) @ V.T
                    # Find w (optimize)
                    wa     = Newton_m(Jp, nvrs, zeros(N), is_inverted=True)
                    # wa   = Newton_m(Jp,Jpp ,zeros(N))
                    # wa   = fmin_bfgs(J,zeros(N),Jp,disp=0)
                    l1     = sqrt(N1/za(wa))

                # Uncomment to revert to ETKF
                # l1 = 1.0

                # Explicitly inflate prior
                # => formulae look different from `bib.bocquet2015expanding`.
                A *= l1
                Y *= l1

                # Compute sqrt update
                Pw = (V * dgn_N(l1)**(-1.0)) @ V.T
                w  = dy@R.inv@Y.T@Pw
                # For the anomalies:
                if not self.Hess:
                    # Regular ETKF (i.e. sym sqrt) update (with inflation)
                    T = (V * dgn_N(l1)**(-0.5)) @ V.T * sqrt(N1)
                    # = (Y@R.inv@Y.T/N1 + eye(N))**(-0.5)
                else:
                    # Also include angular-radial co-dependence.
                    # Note: denominator not squared coz
                    # unlike `bib.bocquet2015expanding` we have inflated Y.
                    Hw = Y@R.inv@Y.T/N1 + eye(N) - 2*np.outer(w, w)/(eN + w@w)
                    T  = funm_psd(Hw, lambda x: x**-.5)  # is there a sqrtm Woodbury?

                E = mu + w@A + T@A
                E = post_process(E, self.infl, self.rot)

                self.stats.infl[ko] = l1
                self.stats.trHK[ko] = (((l1*s)**2 + N1)**(-1.0)*s**2).sum()/HMM.Ny

            self.stats.assess(k, ko, E=E)
