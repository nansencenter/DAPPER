"""Weight- & resampling-based DA methods."""

import numpy as np
import numpy.random as rnd

from dapper.stats import unbias_var, weight_degeneracy
from dapper.tools.linalg import mldiv, mrdiv, pad0, svd0, tinv
from dapper.tools.matrices import chol_reduce, funm_psd
from dapper.tools.progressbar import progbar

from . import da_method


@da_method
class particle_method:
    """Declare default particle arguments."""

    NER: float = 1.0
    resampl: str = 'Sys'


@particle_method
class PartFilt:
    r"""Particle filter ≡ Sequential importance (re)sampling SIS (SIR).

    Refs: `bib.wikle2007bayesian`, `bib.van2009particle`, `bib.chen2003bayesian`

    This is the bootstrap version: the proposal density is just

    $$ q(x_{0:t} \mid y_{1:t}) = p(x_{0:t}) = p(x_t \mid x_{t-1}) p(x_{0:t-1}) $$

    Tuning settings:

     - NER: Trigger resampling whenever `N_eff <= N*NER`.
       If resampling with some variant of 'Multinomial',
       no systematic bias is introduced.
     - qroot: "Inflate" (anneal) the proposal noise kernels
       by this root to increase diversity.
       The weights are updated to maintain un-biased-ness.
       See `bib.chen2003bayesian`, section VI-M.2
    """

    N: int
    reg: float   = 0
    nuj: bool    = True
    qroot: float = 1.0
    wroot: float = 1.0

    # TODO 6:
    # if miN < 1:
    # miN = N*miN

    def assimilate(self, HMM, xx, yy):
        N, Nx, Rm12 = self.N, HMM.Dyn.M, HMM.Obs.noise.C.sym_sqrt_inv

        E = HMM.X0.sample(N)
        w = 1/N*np.ones(N)

        self.stats.assess(0, E=E, w=w)

        for k, ko, t, dt in progbar(HMM.tseq.ticker):
            E = HMM.Dyn(E, t-dt, dt)
            if HMM.Dyn.noise.C != 0:
                D  = rnd.randn(N, Nx)
                E += np.sqrt(dt*self.qroot)*(D@HMM.Dyn.noise.C.Right)

                if self.qroot != 1.0:
                    # Evaluate p/q (for each col of D) when q:=p**(1/self.qroot).
                    w *= np.exp(-0.5*np.sum(D**2, axis=1) * (1 - 1/self.qroot))
                    w /= w.sum()

            if ko is not None:
                self.stats.assess(k, ko, 'f', E=E, w=w)

                innovs = (yy[ko] - HMM.Obs(E, t)) @ Rm12.T
                w      = reweight(w, innovs=innovs)

                if trigger_resampling(w, self.NER, [self.stats, E, k, ko]):
                    C12     = self.reg*auto_bandw(N, Nx)*raw_C12(E, w)
                    # C12  *= np.sqrt(rroot) # Re-include?
                    idx, w  = resample(w, self.resampl, wroot=self.wroot)
                    E, chi2 = regularize(C12, E, idx, self.nuj)
                    # if rroot != 1.0:
                    #     # Compensate for rroot
                    #     w *= np.exp(-0.5*chi2*(1 - 1/rroot))
                    #     w /= w.sum()
            self.stats.assess(k, ko, 'u', E=E, w=w)


@particle_method
class OptPF:
    """'Optimal proposal' particle filter, also known as 'Implicit particle filter'.

    Ref: `bib.bocquet2010beyond`.

    .. note:: Regularization (`Qs`) is here added BEFORE Bayes' rule.
              If `Qs==0`: OptPF should be equal to
              the bootstrap filter :func:`PartFilt`.
    """

    N: int
    Qs: float
    reg: float   = 0
    nuj: bool    = True
    wroot: float = 1.0

    def assimilate(self, HMM, xx, yy):
        N, Nx, R = self.N, HMM.Dyn.M, HMM.Obs.noise.C.full

        E = HMM.X0.sample(N)
        w = 1/N*np.ones(N)

        self.stats.assess(0, E=E, w=w)

        for k, ko, t, dt in progbar(HMM.tseq.ticker):
            E = HMM.Dyn(E, t-dt, dt)
            if HMM.Dyn.noise.C != 0:
                E += np.sqrt(dt)*(rnd.randn(N, Nx)@HMM.Dyn.noise.C.Right)

            if ko is not None:
                self.stats.assess(k, ko, 'f', E=E, w=w)
                y = yy[ko]

                Eo = HMM.Obs(E, t)
                innovs = y - Eo

                # EnKF-ish update
                s   = self.Qs*auto_bandw(N, Nx)
                As  = s*raw_C12(E, w)
                Ys  = s*raw_C12(Eo, w)
                C   = Ys.T@Ys + R
                KG  = As.T@mrdiv(Ys, C)
                E  += sample_quickly_with(As)[0]
                D   = HMM.Obs.noise.sample(N)
                dE  = KG @ (y-HMM.Obs(E, t)-D).T
                E   = E + dE.T

                # Importance weighting
                chi2   = innovs*mldiv(C, innovs.T).T
                logL   = -0.5 * np.sum(chi2, axis=1)
                w      = reweight(w, logL=logL)

                # Resampling
                if trigger_resampling(w, self.NER, [self.stats, E, k, ko]):
                    C12     = self.reg*auto_bandw(N, Nx)*raw_C12(E, w)
                    idx, w  = resample(w, self.resampl, wroot=self.wroot)
                    E, _    = regularize(C12, E, idx, self.nuj)

            self.stats.assess(k, ko, 'u', E=E, w=w)


@particle_method
class PFa:
    """PF with weight adjustment withOUT compensating for the bias it introduces.

    'alpha' sets wroot before resampling such that N_effective becomes >alpha*N.

    Using alpha≈NER usually works well.

    Explanation:
    Recall that the bootstrap particle filter has "no" bias,
    but significant variance (which is reflected in the weights).
    The EnKF is quite the opposite.
    Similarly, by adjusting the weights we play on the bias-variance spectrum.

    NB: This does not mean that we make a PF-EnKF hybrid
    -- we're only playing on the weights.

    Hybridization with xN did not show much promise.
    """

    N: int
    alpha: float
    reg: float   = 0
    nuj: bool    = True
    qroot: float = 1.0

    def assimilate(self, HMM, xx, yy):
        N, Nx, Rm12 = self.N, HMM.Dyn.M, HMM.Obs.noise.C.sym_sqrt_inv

        E = HMM.X0.sample(N)
        w = 1/N*np.ones(N)

        self.stats.assess(0, E=E, w=w)

        for k, ko, t, dt in progbar(HMM.tseq.ticker):
            E = HMM.Dyn(E, t-dt, dt)
            if HMM.Dyn.noise.C != 0:
                D  = rnd.randn(N, Nx)
                E += np.sqrt(dt*self.qroot)*(D@HMM.Dyn.noise.C.Right)

                if self.qroot != 1.0:
                    # Evaluate p/q (for each col of D) when q:=p**(1/self.qroot).
                    w *= np.exp(-0.5*np.sum(D**2, axis=1) * (1 - 1/self.qroot))
                    w /= w.sum()

            if ko is not None:
                self.stats.assess(k, ko, 'f', E=E, w=w)

                innovs = (yy[ko] - HMM.Obs(E, t)) @ Rm12.T
                w      = reweight(w, innovs=innovs)

                if trigger_resampling(w, self.NER, [self.stats, E, k, ko]):
                    C12    = self.reg*auto_bandw(N, Nx)*raw_C12(E, w)
                    # C12  *= np.sqrt(rroot) # Re-include?

                    wroot = 1.0
                    while True:
                        s   = (w**(1/wroot - 1)).clip(max=1e100)
                        s  /= (s*w).sum()
                        sw  = s*w
                        if 1/(sw@sw) < N*self.alpha:
                            wroot += 0.2
                        else:
                            self.stats.wroot[ko] = wroot
                            break
                    idx, w  = resample(sw, self.resampl, wroot=1)

                    E, chi2 = regularize(C12, E, idx, self.nuj)
                    # if rroot != 1.0:
                    #     Compensate for rroot
                    #     w *= np.exp(-0.5*chi2*(1 - 1/rroot))
                    #     w /= w.sum()
            self.stats.assess(k, ko, 'u', E=E, w=w)


@particle_method
class PFxN_EnKF:
    """Particle filter with EnKF-based proposal, q.

    Also employs xN duplication, as in PFxN.

    Recall that the proposals:
    Opt.: q_n(x) = c_n·N(x|x_n,Q     )·N(y|Hx,R)  (1)
    EnKF: q_n(x) = c_n·N(x|x_n,bar{B})·N(y|Hx,R)  (2)
    with c_n = p(y|x^{k-1}_n) being the composite proposal-analysis weight,
    and with Q possibly from regularization (rather than actual model noise).

    Here, we will use the posterior mean of (2) and cov of (1).
    Or maybe we should use x_a^n distributed according to a sqrt update?
    """

    N: int
    Qs: float
    xN: int
    re_use: bool = True
    wroot_max: float = 5.0

    def assimilate(self, HMM, xx, yy):
        N, xN, Nx  = self.N, self.xN, HMM.Dyn.M
        Rm12, Ri = HMM.Obs.noise.C.sym_sqrt_inv, HMM.Obs.noise.C.inv

        E = HMM.X0.sample(N)
        w = 1/N*np.ones(N)

        DD = None

        self.stats.assess(0, E=E, w=w)

        for k, ko, t, dt in progbar(HMM.tseq.ticker):
            E = HMM.Dyn(E, t-dt, dt)
            if HMM.Dyn.noise.C != 0:
                E += np.sqrt(dt)*(rnd.randn(N, Nx)@HMM.Dyn.noise.C.Right)

            if ko is not None:
                self.stats.assess(k, ko, 'f', E=E, w=w)
                y  = yy[ko]
                Eo = HMM.Obs(E, t)
                wD = w.copy()

                # Importance weighting
                innovs = (y - Eo) @ Rm12.T
                w      = reweight(w, innovs=innovs)

                # Resampling
                if trigger_resampling(w, self.NER, [self.stats, E, k, ko]):
                    # Weighted covariance factors
                    Aw = raw_C12(E, wD)
                    Yw = raw_C12(Eo, wD)

                    # EnKF-without-pertubations update
                    if N > Nx:
                        C       = Yw.T @ Yw + HMM.Obs.noise.C.full
                        KG      = mrdiv(Aw.T@Yw, C)
                        cntrs   = E + (y-Eo)@KG.T
                        Pa      = Aw.T@Aw - KG@Yw.T@Aw
                        P_cholU = funm_psd(Pa, np.sqrt)
                        if DD is None or not self.re_use:
                            DD    = rnd.randn(N*xN, Nx)
                            chi2  = np.sum(DD**2, axis=1) * Nx/N
                            log_q = -0.5 * chi2
                    else:
                        V, sig, UT = svd0(Yw @ Rm12.T)
                        dgn      = pad0(sig**2, N) + 1
                        Pw       = (V * dgn**(-1.0)) @ V.T
                        cntrs    = E + (y-Eo)@Ri@Yw.T@Pw@Aw
                        P_cholU  = (V*dgn**(-0.5)).T @ Aw
                        # Generate N·xN random numbers from NormDist(0,1),
                        # and compute log(q(x))
                        if DD is None or not self.re_use:
                            rnk   = min(Nx, N-1)
                            DD    = rnd.randn(N*xN, N)
                            chi2  = np.sum(DD**2, axis=1) * rnk/N
                            log_q = -0.5 * chi2
                        # NB: the DoF_linalg/DoF_stoch correction
                        # is only correct "on average".
                        # It is inexact "in proportion" to V@V.T-Id,
                        # where V,s,UT = tsvd(Aw).
                        # Anyways, we're computing the tsvd of Aw below,
                        # so might as well compute q(x) instead of q(xi).

                    # Duplicate
                    ED  = cntrs.repeat(xN, 0)
                    wD  = wD.repeat(xN) / xN

                    # Sample q
                    AD = DD@P_cholU
                    ED = ED + AD

                    # log(prior_kernel(x))
                    s         = self.Qs*auto_bandw(N, Nx)
                    innovs_pf = AD @ tinv(s*Aw)
                    # NB: Correct: innovs_pf = (ED-E_orig) @ tinv(s*Aw)
                    #     But it seems to make no difference on well-tuned performance !
                    log_pf    = -0.5 * np.sum(innovs_pf**2, axis=1)

                    # log(likelihood(x))
                    innovs = (y - HMM.Obs(ED, t)) @ Rm12.T
                    log_L  = -0.5 * np.sum(innovs**2, axis=1)

                    # Update weights
                    log_tot = log_L + log_pf - log_q
                    wD      = reweight(wD, logL=log_tot)

                    # Resample and reduce
                    wroot = 1.0
                    while wroot < self.wroot_max:
                        idx, w = resample(wD, self.resampl, wroot=wroot, N=N)
                        dups   = sum(mask_unique_of_sorted(idx))
                        if dups == 0:
                            E = ED[idx]
                            break
                        else:
                            wroot += 0.1
            self.stats.assess(k, ko, 'u', E=E, w=w)


@particle_method
class PFxN:
    """Particle filter with buckshot duplication during analysis.

    Idea: sample xN duplicates from each of the N kernels.
    Let resampling reduce it to N.

    Additional idea: employ w-adjustment to obtain N unique particles,
    without jittering.
    """

    N: int
    Qs: float
    xN: int
    re_use: bool = True
    wroot_max: float = 5.0

    def assimilate(self, HMM, xx, yy):
        N, xN, Nx, Rm12 = self.N, self.xN, HMM.Dyn.M, HMM.Obs.noise.C.sym_sqrt_inv

        DD = None
        E  = HMM.X0.sample(N)
        w  = 1/N*np.ones(N)

        self.stats.assess(0, E=E, w=w)

        for k, ko, t, dt in progbar(HMM.tseq.ticker):
            E = HMM.Dyn(E, t-dt, dt)
            if HMM.Dyn.noise.C != 0:
                E += np.sqrt(dt)*(rnd.randn(N, Nx)@HMM.Dyn.noise.C.Right)

            if ko is not None:
                self.stats.assess(k, ko, 'f', E=E, w=w)
                y  = yy[ko]
                wD = w.copy()

                innovs = (y - HMM.Obs(E, t)) @ Rm12.T
                w      = reweight(w, innovs=innovs)

                if trigger_resampling(w, self.NER, [self.stats, E, k, ko]):
                    # Compute kernel colouring matrix
                    cholR = self.Qs*auto_bandw(N, Nx)*raw_C12(E, wD)
                    cholR = chol_reduce(cholR)

                    # Generate N·xN random numbers from NormDist(0,1)
                    if DD is None or not self.re_use:
                        DD = rnd.randn(N*xN, Nx)

                    # Duplicate and jitter
                    ED  = E.repeat(xN, 0)
                    wD  = wD.repeat(xN) / xN
                    ED += DD[:, :len(cholR)]@cholR

                    # Update weights
                    innovs = (y - HMM.Obs(ED, t)) @ Rm12.T
                    wD     = reweight(wD, innovs=innovs)

                    # Resample and reduce
                    wroot = 1.0
                    while wroot < self.wroot_max:
                        idx, w = resample(wD, self.resampl, wroot=wroot, N=N)
                        dups   = sum(mask_unique_of_sorted(idx))
                        if dups == 0:
                            E = ED[idx]
                            break
                        else:
                            wroot += 0.1
            self.stats.assess(k, ko, 'u', E=E, w=w)


def trigger_resampling(w, NER, stat_args):
    """Return boolean: N_effective <= threshold. Also write self.stats."""
    N_eff       = 1/(w@w)
    do_resample = N_eff <= len(w)*NER

    # Unpack stat args
    stats, E, k, ko = stat_args

    stats.N_eff[ko]  = N_eff
    stats.resmpl[ko] = 1 if do_resample else 0

    # Why have we put self.stats.assess() here?
    # Because we need to write self.stats.N_eff and self.stats.resmpl before calling
    # assess() so that these curves (in sliding_diagnostics liveplotting
    # are not eliminated (as inactive).
    stats.assess(k, ko, 'a', E=E, w=w)

    return do_resample


def all_but_1_is_None(*args):
    """Check if only 1 of the items in list are Truthy."""
    return sum(x is not None for x in args) == 1


def reweight(w, lklhd=None, logL=None, innovs=None):
    r"""Do Bayes' rule (for the empirical distribution of an importance sample).

    Do computations in log-space, for at least 2 reasons:

    - Normalization: will fail if `sum==0` (if all innov's are large).
    - Num. precision: `lklhd*w` should have better precision in log space.

    Output is non-log, for the purpose of assessment and resampling.

    If input is 'innovs', then
    $$\text{likelihood} = \mathcal{N}(\text{innovs}|0,I)$$.
    """
    assert all_but_1_is_None(lklhd, logL, innovs), \
        "Input error. Only specify one of lklhd, logL, innovs"

    # Get log-values.
    # Use context manager 'errstate' to not warn for log(0) = -inf.
    # Note: the case when all(w==0) will cause nan's,
    #       which should cause errors outside.
    with np.errstate(divide='ignore'):
        logw = np.log(w)
        if lklhd is not None:
            logL = np.log(lklhd)
        elif innovs is not None:
            chi2 = np.sum(innovs**2, axis=1)
            logL = -0.5 * chi2

    logw   = logw + logL   # Bayes' rule in log-space
    logw  -= logw.max()    # Avoid numerical error
    w      = np.exp(logw)  # non-log
    w     /= w.sum()       # normalize
    return w


def raw_C12(E, w):
    """Compute the 'raw' matrix-square-root of the ensemble' covariance.

    The weights are used both for the mean and anomalies (raw sqrt).

    Note: anomalies (and thus cov) are weighted,
    and also computed based on a weighted mean.
    """
    # If weights are degenerate: use unweighted covariance to avoid C=0.
    if weight_degeneracy(w):
        w = np.ones(len(w))/len(w)
        # PS: 'avoid_pathological' already treated here.

    mu  = w@E
    A   = E - mu
    ub  = unbias_var(w, avoid_pathological=False)
    C12 = np.sqrt(ub*w[:, None]) * A
    return C12


def mask_unique_of_sorted(idx):
    """Find unique values assuming `idx` is sorted.

    NB: returns a mask which is `True` at `[i]` iff `idx[i]` is *not* unique.
    """
    duplicates  = idx == np.roll(idx, 1)
    duplicates |= idx == np.roll(idx, -1)
    return duplicates


def auto_bandw(N, M):
    """Optimal bandwidth (not bandwidth^2), as per Scott's rule-of-thumb.

    Refs: `bib.doucet2001sequential` section 12.2.2, [Wik17]_ section "Rule_of_thumb"
    """
    return N**(-1/(M+4))


def regularize(C12, E, idx, no_uniq_jitter):
    """Jitter (add noise).

    After resampling some of the particles will be identical.
    Therefore, if noise.is_deterministic: some noise must be added.
    This is adjusted by the regularization 'reg' factor
    (so-named because Dirac-deltas are approximated  Gaussian kernels),
    which controls the strength of the jitter.
    This causes a bias. But, as N-->∞, the reg. bandwidth-->0, i.e. bias-->0.
    Ref: `bib.doucet2001sequential`, section 12.2.2.
    """
    # Select
    E = E[idx]

    # Jitter
    if no_uniq_jitter:
        dups         = mask_unique_of_sorted(idx)
        sample, chi2 = sample_quickly_with(C12, N=sum(dups))
        E[dups]     += sample
    else:
        sample, chi2 = sample_quickly_with(C12, N=len(E))
        E           += sample

    return E, chi2


def resample(w, kind='Systematic', N=None, wroot=1.0):
    """Multinomial resampling.

    Refs: `bib.doucet2009tutorial`, `bib.van2009particle`, `bib.liu2001theoretical`.

    - kind: 'Systematic', 'Residual' or 'Stochastic'.
      'Stochastic' corresponds to `rnd.choice` or `rnd.multinomial`.
      'Systematic' and 'Residual' are more systematic (less stochastic)
      varaitions of 'Stochastic' sampling.
      Among the three, 'Systematic' is fastest, introduces the least noise,
      and brings continuity benefits for localized particle filters,
      and is therefore generally prefered.
      Example: see docs/snippets/ex_resample.py.

    - N can be different from len(w)
      (e.g. in case some particles have been elimintated).

    - wroot: Adjust weights before resampling by this root to
      promote particle diversity and mitigate thinning.
      The outcomes of the resampling are then weighted to maintain un-biased-ness.
      Ref: `bib.liu2001theoretical`, section 3.1

    Note: (a) resampling methods are beneficial because they discard
    low-weight ("doomed") particles and reduce the variance of the weights.
    However, (b) even unbiased/rigorous resampling methods introduce noise;
    (increases the var of any empirical estimator, see [1], section 3.4).
    How to unify the seemingly contrary statements of (a) and (b) ?
    By recognizing that we're in the *sequential/dynamical* setting,
    and that *future* variance may be expected to be lower by focusing
    on the high-weight particles which we anticipate will
    have more informative (and less variable) future likelihoods.
    """
    assert(abs(w.sum()-1) < 1e-5)

    # Input parsing
    N_o = len(w)   # N _original
    if N is None:  # N to sample
        N = N_o

    # Compute factors s such that s*w := w**(1/wroot).
    if wroot != 1.0:
        s   = (w**(1/wroot - 1)).clip(max=1e100)
        s  /= (s*w).sum()
        sw  = s*w
    else:
        s   = np.ones(N_o)
        sw  = w

    # Do the actual resampling
    idx = _resample(sw, kind, N_o, N)

    w  = 1/s[idx]  # compensate for above scaling by s
    w /= w.sum()   # normalize

    return idx, w


def _resample(w, kind, N_o, N):
    """Core functionality for :func:`resample`."""
    if kind in ['Stochastic', 'Stoch']:
        # van Leeuwen [2] also calls this "probabilistic" resampling
        idx = rnd.choice(N_o, N, replace=True, p=w)
        # rnd.multinomial is faster (slightly different usage) ?
    elif kind in ['Residual', 'Res']:
        # Doucet [1] also calls this "stratified" resampling.
        w_N   = w*N              # upscale
        w_I   = w_N.astype(int)  # integer part
        w_D   = w_N-w_I          # decimal part
        # Create duplicate indices for integer parts
        idx_I = [i*np.ones(wi, dtype=int) for i, wi in enumerate(w_I)]
        idx_I = np.concatenate(idx_I)
        # Multinomial sampling of decimal parts
        N_I   = w_I.sum()  # == len(idx_I)
        N_D   = N - N_I
        idx_D = rnd.choice(N_o, N_D, replace=True, p=w_D/w_D.sum())
        # Concatenate
        idx   = np.hstack((idx_I, idx_D))
    elif kind in ['Systematic', 'Sys']:
        # van Leeuwen [2] also calls this "stochastic universal" resampling
        U     = rnd.rand(1) / N
        CDF_a = U + np.arange(N)/N
        CDF_o = np.cumsum(w)
        # idx = CDF_a <= CDF_o[:,None]
        # idx = np.argmax(idx,axis=0) # Finds 1st. SO/a/16244044/
        idx   = np.searchsorted(CDF_o, CDF_a)
    else:
        raise KeyError
    return idx


def sample_quickly_with(C12, N=None):
    """Gaussian sampling in the quickest fashion.

    Method depends on the size of the colouring matrix `C12`.
    """
    (N_, M) = C12.shape
    if N is None:
        N = N_
    if N_ > 2*M:
        cholR  = chol_reduce(C12)
        D      = rnd.randn(N, cholR.shape[0])
        chi2   = np.sum(D**2, axis=1)
        sample = D@cholR
    else:
        chi2_compensate_for_rank = min(M/N_, 1.0)
        D      = rnd.randn(N, N_)
        chi2   = np.sum(D**2, axis=1) * chi2_compensate_for_rank
        sample = D@C12
    return sample, chi2
