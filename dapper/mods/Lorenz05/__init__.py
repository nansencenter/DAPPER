"""A multi-scale, smooth version of the classic Lorenz-96.

This is an implementation of "Model III" of `bib.lorenz2005designing`.

Similar to `dapper.mods.LorenzUV` this model is designed
to contain two different scales. However, in "Model III"
the two scales are not kept separate, but superimposed,
and the large scale variables are (adjustably) spatially smooth.

Interestingly, the model is known as "Lorenz 04" in DART, where it was
coded by Hansen (colleague of Lorenz) in 2004 (prior to publication).

Special cases of this model are:

- Set `J=1` to get "Model II".
- Set `K=1` (and `J=1`) to get "Model I",
  which is the same as the Lorenz-96 model.

An implementation using explicit for-loops can be found in commit 6193532b .
It uses numba (pip install required) for speed gain, but is still very slow.
The implementation hereunder uses efficient numpy vectorization => much faster.

With rk4 the largest stable time step (for free run) seems to be
somewhere around what Lorenz used, namely `dt=0.05/12`.
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import convolve1d

from dapper.mods.integration import rk4


@dataclass
class Model:
    """The model configuration.

    Functionality that does not directly depend on the model parameters
    has been left outside of this class.

    Using OOP (rather than module-level encapsulation) facilitates
    working with multiple parameter settings simultaneously.
    """
    M    : int   = 960  # state vector length
    J    : int   = 12   # decomposition kernel radius (width/2)
    K    : int   = 32   # smoothing kernel width (increase for +smooth and -waves)
    b    : float = 10   # scaling of small-scale variability
    c    : float = 2.5  # coupling strength
    Force: float = 15   # forcing

    mp   : bool  = False

    def __post_init__(self):
        J = self.J
        self.alpha = (3*J**2 + 3) / (2*J**3 + 4*J)
        self.beta  = (2*J**2 + 1) / (J**4 + 2*J**2)

        # Heuristic
        self.x0 = 3 + 4*np.ones(self.M)
        self.x0[0] += 1  # perturb from stable point

    def decompose(self, z):
        """Split `z` into `x` and `y` fields, where `x` is the large-scale component."""
        width = 2*self.J  # not +1 coz the sum in Lorenz eqn 13a is never ordinary
        _, weights, inds0 = summation_kernel(width)
        weights *= self.alpha - self.beta*abs(inds0)
        x = convolve1d(z, weights, mode="wrap")
        # Manual implementation:
        # x = np.zeros_like(z)
        # for m in range(M):
        #     for i, w in zip(inds0, weights):
        #         x[..., m] += w * z[..., mod(m + i)]
        y = z - x
        return x, y

    def dxdt(self, z):
        x, y = self.decompose(z)

        return (
            + prodsum_self(x, self.K)       # "convection" of x
            + prodsum_K1(y, y) * self.b**2  # "convection" of y
            + prodsum_K1(y, x) * self.c     # coupling
            + -x - y*self.b                 # damping
            + self.Force
        )

    def step1(self, x0, t, dt):
        return rk4(lambda x, t: self.dxdt(x), x0, np.nan, dt)
    # Note: step1 is already ensemble compatible.
    # However, it is a bit slow, so we can gain in speed by using multiprocessing.

    # TODO 4: the same pattern is used for the QG model. Merge, and  it a decorator?
    def step(self, E, t, dt):
        if E.ndim == 1:
            return self.step1(E, t, dt)
        if E.ndim == 2:
            if self.mp and E.size > 1e5:
                import dapper.tools.multiproc as multiproc
                with multiproc.Pool(self.mp) as pool:
                    E = pool.map(lambda x: self.step1(x, t=t, dt=dt), E)
                E = np.array(E)
            else:
                for n, x in enumerate(E):
                    E[n] = self.step1(x, t, dt)
            return E


def summation_kernel(width):
    """Prepare computation of the modified sum in `bib.lorenz2005designing`.

    Note: This gets repeatedly called, but actually the input is only ever
    `width = K` or `2*J`, so we should really cache or pre-compute.
    But with default system parameters and N=50, the savings are negligible.
    """
    r = width // 2  # "radius"
    weights = np.ones(2*r + 1)
    if width != len(weights):
        weights[0] = weights[-1] = .5
    inds0 = np.arange(-r, r+1)
    return r, weights, inds0


def boxcar(x, n, method="direct"):
    """Moving average (boxcar filter) on `x` using `n` nearest (periodically) elements.

    For symmetry, if `n` is pair, the actual number of elements used is `n+1`,
    and the outer elements weighted by `0.5` to compensate for the `+1`.

    This is the modified sum of `bib.lorenz2005designing`, used e.g. in eqn. 9.
    For intuition: this type of summation (and the associated Sigma prime notation)
    may also be found for the "Trapezoidal rule" and in the inverse DFT used in
    spectral methods on a periodic domain.

    Apart from this weighting, this constitutes merely a boxcar filter.
    There are of course several well-known implementations.  The computational
    suggestion suggested by Lorenz below eqn 10 could maybe be implemented
    (with vectorisation) using `cumsum`, but this seems tricky due to weighting
    and periodicity.

    [1](https://stackoverflow.com/q/14313510)
    [2](https://stackoverflow.com/q/13728392)
    [3](https://stackoverflow.com/a/38034801)

    In testing with default system parameters, and ensemble size N=50, the
    "direct" method is generally 2x faster than the "fft" method, and the "oa"
    method is a little slower again. If `K` or `J` is increased, then the "fft"
    method becomes the fastest.

    Examples:
    >>> x = np.array([0, 1, 2], dtype=float)
    >>> np.allclose(boxcar(x, 1), x)
    True
    >>> boxcar(x, 2)
    array([0.75, 1.  , 1.25])
    >>> boxcar(x, 3)
    array([1., 1., 1.])
    >>> x = np.arange(10, dtype=float)
    >>> boxcar(x, 2)
    array([2.5, 1. , 2. , 3. , 4. , 5. , 6. , 7. , 8. , 6.5])
    >>> boxcar(x, 5)
    array([4., 3., 2., 3., 4., 5., 6., 7., 6., 5.])
    """
    r, weights, inds0 = summation_kernel(n)
    M = x.shape[-1]

    if method == "manual":
        def mod(ind):
            return np.mod(ind, M)
        a = np.zeros_like(x)
        for m in range(M):
            a[..., m] = x[..., mod(m + inds0)] @ weights
            # for i, w in zip(inds0, weights):
            #     a[..., m] += x[..., mod(m + i)] * w

    elif method in ["fft", "oa"]:
        # - Requires wrapping the state vector for periodicity.
        #   Maybe this could be avoided if we do the fft ourselves?
        # - `np.convolve` does not support multi-dim arrays (for ensembles).
        # - `ss.convolve` either does the computation "directly" itself,
        #   or delegates the job to ss.fftconvolve or ss.oaconvolve.
        #   Strangely, only the latter subroutines support the axis argument
        #   so we must call them ourselves.
        if method == "fft":
            from scipy.signal import fftconvolve as convolver
        else:
            from scipy.signal import oaconvolve as convolver
        weights = weights[... if x.ndim == 1 else None]  # dim compatibility
        xxx = np.hstack([x[..., -r:], x, x[..., :r]])  # wrap
        a = convolver(xxx, weights, axes=-1)
        a = a[..., 2*r:-2*r]  # Trim (rm wrapped edges)

    else:  # method == "direct":
        # AFAICT, this uses "direct" computations.
        a = convolve1d(x, weights, mode="wrap")

    a /= n
    return a


def shift(x, k):
    """Rolls `x` leftwards. I.e. `output[i] = input[i+k]`.

    Notes about speed that usually hold when testing with ensemble DA:
    - This implementation is somewhat faster than `x[..., np.mod(ii + k, M)]`.
    - Computational savings of re-using already shifted vectors (or matrices)
      compared to just calling this function again are negligible.
    """
    return np.roll(x, -k, axis=-1)


def prodsum_self(x, k):
    """Compute `prodsum(x, x, k)` efficiently: eqn 10 of `bib.lorenz2005designing`."""
    W = boxcar(x, k)
    WW = shift(W, -2*k) * shift(W, -k)
    WX = shift(W, -k) * shift(x, k)
    WX = boxcar(WX, k)
    return - WW + WX


def prodsum_K1(x, y):
    """Compute `prodsum(x, y, 1)` efficiently."""
    return -shift(x, -2) * shift(y, -1) + shift(x, -1) * shift(y, +1)
