"""Linear algebra."""

import numpy as np
import scipy.linalg as sla


def mrdiv(b, A):
    """b/A."""
    return sla.solve(A.T, b.T).T


def mldiv(A, b):
    r"""A\b."""
    return sla.solve(A, b)


def truncate_rank(s, threshold, avoid_pathological):
    """Find `r` such that `s[:r]` contains the threshold proportion of `s`."""
    assert isinstance(threshold, float)
    if threshold == 1.0:
        r = len(s)
    elif threshold < 1.0:
        r = np.sum(np.cumsum(s)/np.sum(s) < threshold)
        r += 1  # Hence the strict inequality above
        if avoid_pathological:
            # If not avoid_pathological, then the last 4 diag. entries of
            # svdi( *tsvd(np.eye(400),0.99) )
            # will be zero. This is probably not intended.
            r += np.sum(np.isclose(s[r-1], s[r:]))
    else:
        raise ValueError
    return r


def tsvd(A, threshold=0.99999, avoid_pathological=True):
    """Compute the truncated svd.

    Also automates 'full_matrices' flag.

    Parameters
    ----------
    avoid_pathological: bool
        Avoid truncating (e.g.) the identity matrix.
        NB: only applies for float threshold.

    threshold: float or int

    - if float, < 1.0 then "rank" = lowest number
      such that the "energy" retained >= threshold
    - if int,  >= 1   then "rank" = threshold
    """
    M, N = A.shape
    full_matrices = False

    if isinstance(threshold, int):
        # Assume specific number is requested
        r = threshold
        assert 1 <= r <= max(M, N)
        if r > min(M, N):
            full_matrices = True
            r = min(M, N)

    U, s, VT = sla.svd(A, full_matrices)

    if isinstance(threshold, float):
        # Assume proportion is requested
        r = truncate_rank(s, threshold, avoid_pathological)

    # Truncate
    U = U[:, :r]
    VT = VT[:r]
    s = s[:r]
    return U, s, VT


def svd0(A):
    """Similar to Matlab's `svd(A,0)`.

    Compute the

    - full    svd if `nrows > ncols`
    - reduced svd otherwise.

    As in Matlab: `svd(A,0)`,
    except that the input and output are transposed, in keeping with DAPPER convention.
    It contrasts with `scipy.linalg.svd(full_matrice=False)`
    and Matlab's `svd(A,'econ')`, both of which always compute the reduced svd.


    See Also
    --------
    tsvd : rank (and threshold) truncation.
    """
    M, N = A.shape
    if M > N:
        return sla.svd(A, full_matrices=True)
    return sla.svd(A, full_matrices=False)


def pad0(x, N):
    """Pad `x` with zeros so that `len(x)==N`."""
    out = np.zeros(N)
    out[:len(x)] = x
    return out


def svdi(U, s, VT):
    """Reconstruct matrix from `sla.svd` or `tsvd`.

    Examples
    --------
    >>> A = np.arange(12).reshape((3,-1))
    >>> B = svdi(*tsvd(A, 1.0))
    >>> np.allclose(A, B)
    True

    See Also
    --------
    sla.diagsvd
    """
    return (U[:, :len(s)] * s) @ VT


def tinv(A, *kargs, **kwargs):
    """Psuedo-inverse using `tsvd`.

    See Also
    --------
    sla.pinv2.
    """
    U, s, VT = tsvd(A, *kargs, **kwargs)
    return (VT.T * s**(-1.0)) @ U.T


def trank(A, *kargs, **kwargs):
    """Compute rank via `tsvd`, i.e. as "seen" by `tsvd`."""
    return len(tsvd(A, *kargs, **kwargs)[1])
