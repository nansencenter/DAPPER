"""Localization tools, including distance and tapering comps.

A good introduction to localization:
Sakov (2011), Computational Geosciences:
'Relation between two common localisation methods for the EnKF'.
"""

import itertools

import numpy as np

# Defaults
CUTOFF   = 1e-3
TAG      = 'GC'
PERIODIC = True


def pairwise_distances(A, B=None, periodic=PERIODIC, domain=None):
    """Euclidian distance (un-squared) between pts in A and B.

    - A: 2-dim array of shape (npts,ndim).
         If A is 1-dim, then (ndim)<--A.shape, npts<--(,).
    - B: same as for A (npts can differ).
    - returns distances as an array of shape (npts_A, npts_B)

    These are equal:
    >>> pairwise_distances(A,periodic=False)
    >>> squareform(pdist(A)) # 2x slower

    But pairwise_distances allows comparing two different pts A to pts B
    (without the augmentation/block tricks needed for pdist).

    Also, the ``periodic`` option is provided.
    It assumes the domain is a hyper-rectangle with edge lengths: ``domain``.
    NB: Behaviour not defined for any(A.max(0) > domain), and likewise for B.
    """

    if B is None:
        B = A

    # Prep A,B
    A = np.asarray(A)
    B = np.asarray(B)
    npts_A = A.shape[:A.ndim-1]
    npts_B = B.shape[:B.ndim-1]
    # assert A.shape[-1] == B.shape[-1] # == ndim
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)

    d = A[:, None] - B  # shape=(npts_A,npts_B,ndim)

    if periodic:
        domain = np.reshape(domain, (1, 1, -1))  # for broadcasting
        d = abs(d)
        d = np.minimum(d, domain-d)

    # scipy.linalg.norm(d,axis=-1)
    distances = np.sqrt((d**2).sum(axis=-1))

    return distances.reshape(npts_A+npts_B)


def dist2coeff(dists, radius, tag=None):
    """Compute tapering coefficients corresponding to a distances.

    NB: The radius is internally adjusted such that, independently of 'tag',
    coeff==np.exp(-0.5) when distance==radius.

    This is largely based on Sakov's enkf-matlab code. Two bugs have here been fixed:
     - The constants were slightly wrong, as noted in comments below.
     - It forgot to take sqrt() of coeffs when applying them through 'local analysis'.
    """
    coeffs = np.zeros(dists.shape)

    if tag is None:
        tag = TAG

    if tag == 'Gauss':
        R = radius
        coeffs = np.exp(-0.5 * (dists/R)**2)
    elif tag == 'Exp':
        R = radius
        coeffs = np.exp(-0.5 * (dists/R)**3)
    elif tag == 'Cubic':
        R            = radius * 1.87  # Sakov: 1.8676
        inds         = dists <= R
        coeffs[inds] = (1 - (dists[inds] / R) ** 3) ** 3
    elif tag == 'Quadro':
        R            = radius * 1.64  # Sakov: 1.7080
        inds         = dists <= R
        coeffs[inds] = (1 - (dists[inds] / R) ** 4) ** 4
    elif tag == 'GC':  # eqn 4.10 of Gaspari-Cohn'99, or eqn 25 of Sakov2011relation
        R = radius * 1.82  # =np.sqrt(10/3). Sakov: 1.7386
        # 1st segment
        ind1         = dists <= R
        r2           = (dists[ind1] / R) ** 2
        r3           = (dists[ind1] / R) ** 3
        coeffs[ind1] = 1 + r2 * (- r3 / 4 + r2 / 2) + r3 * (5 / 8) - r2 * (5 / 3)
        # 2nd segment
        ind2         = np.logical_and(R < dists, dists <= 2*R)
        r1           = (dists[ind2] / R)
        r2           = (dists[ind2] / R) ** 2
        r3           = (dists[ind2] / R) ** 3
        coeffs[ind2] = r2 * (r3 / 12 - r2 / 2) + r3 * (5 / 8) \
            + r2 * (5 / 3) - r1 * 5 + 4 - (2 / 3) / r1
    elif tag == 'Step':
        R            = radius
        inds         = dists <= R
        coeffs[inds] = 1
    else:
        raise KeyError('No such coeff function.')

    return coeffs


def inds_and_coeffs(dists, radius, cutoff=None, tag=None):
    """Returns
    inds   : the indices of pts that are "close to" centre.
    coeffs : the corresponding tapering coefficients.
    """
    if cutoff is None:
        cutoff = CUTOFF

    coeffs = dist2coeff(dists, radius, tag)

    # Truncate using cut-off
    inds   = np.arange(len(dists))[coeffs > cutoff]
    coeffs = coeffs[inds]

    return inds, coeffs


def localization_setup(y2x_distances, batches):

    def localization_now(radius, direction, t, tag=None):
        "Provide localization setup for time t."
        y2x = y2x_distances(t)

        if direction == 'x2y':
            def obs_taperer(batch):
                # Don't use ``batch = batches[iBatch]``
                # (with iBatch as this function's input).
                # This would slow down multiproc.,
                # coz batches gets copied to each process.
                x2y = y2x.T
                dists = x2y[batch].mean(axis=0)
                return inds_and_coeffs(dists, radius, tag=tag)
            return batches, obs_taperer

        elif direction == 'y2x':
            def state_taperer(iObs):
                return inds_and_coeffs(y2x[iObs], radius, tag=tag)
            return state_taperer

    return localization_now


def no_localization(Nx, Ny):

    def obs_taperer(batch):
        return np.arange(Ny), np.ones(Ny)

    def state_taperer(iObs):
        return np.arange(Nx), np.ones(Nx)

    def localization_now(radius, direction, t, tag=None):
        """Returns all indices, with all tapering coeffs=1.

        Used to validate local DA methods [eg LETKF<==>EnKF('Sqrt')]."""
        assert radius == np.inf, "Localization functions not specified"

        if direction == 'x2y':
            return [np.arange(Nx)], obs_taperer
        elif direction == 'y2x':
            return state_taperer

    return localization_now


def rectangular_partitioning(shape, steps, do_ind=True):
    """N-D rectangular batch generation.

    shape: [len(grid[dim]) for dim in range(ndim)]
    steps: [step_len[dim]  for dim in range(ndim)]

    returns a list of batches,
    where each element (batch) is a list of indices

    # Example, with visualization:
    >>> shape   = [4,13]
    >>> steps   = [2,4]
    >>> batches = rectangular_partitioning(shape, steps, do_ind=False)
    >>> M       = np.prod(shape)
    >>> nB      = len(batches)
    >>> values  = np.random.choice(arange(nB),nB,0)
    >>> Z       = np.zeros(shape)
    >>> for ib,b in enumerate(batches):
    >>>   Z[tuple(b)] = values[ib]
    >>> plt.imshow(Z)
    """
    assert len(shape) == len(steps)
    # ndim = len(steps)

    # An ndim list of (average) local grid lengths:
    nLocs = [round(n/d) for n, d in zip(shape, steps)]
    # An ndim list of (marginal) grid partitions
    # [array_split() handles non-divisibility]:
    edge_partitions = [np.array_split(np.arange(n), nLoc)
                       for n, nLoc in zip(shape, nLocs)]

    batches = []
    for batch_edges in itertools.product(*edge_partitions):
        # The 'indexing' argument below is actually inconsequential:
        # it merely changes batch's internal ordering.
        batch_rect  = np.meshgrid(*batch_edges, indexing='ij')
        coords      = [ii.flatten() for ii in batch_rect]
        batches    += [coords]

    if do_ind:
        def sub2ind(sub):
            return np.ravel_multi_index(sub, shape)
        batches = [sub2ind(b) for b in batches]

    return batches


# NB: Don't try to put the time-dependence of obs_inds inside obs_taperer().
# That would require calling ind2sub len(batches) times per analysis,
# and the result cannot be easily cached, because of multiprocessing.
def safe_eval(fun, t):
    try:
        return fun(t)
    except TypeError:
        return fun


# NB: Why is the 'order' argument not supported by this module? Because:
#  1) Assuming only order (orientation) 'C' simplifies the module's code.
#  2) It's not necessary, because the module only communicates to *exterior* via indices
#     [of what assumes to be X.flatten(order='C')], and not coordinates!
#     Thus, the only adaptation necessary if the order is 'F' is to reverse
#     the shape parameter passed to these functions (example: mods/QG/sakov2008).


def nd_Id_localization(shape,
                       batch_shape=None,
                       obs_inds=None,
                       periodic=PERIODIC):
    """Localize Id (direct) point obs of an
    N-D, homogeneous, rectangular domain."""

    M = np.prod(shape)

    if batch_shape is None:
        batch_shape = (1,)*len(shape)
    if obs_inds is None:
        obs_inds = np.arange(M)

    def ind2sub(ind):
        return np.asarray(np.unravel_index(ind, shape)).T

    batches = rectangular_partitioning(shape, batch_shape)

    state_coord = ind2sub(np.arange(M))

    def y2x_distances(t):
        obs_coord = ind2sub(safe_eval(obs_inds, t))
        return pairwise_distances(obs_coord, state_coord, periodic, shape)

    return localization_setup(y2x_distances, batches)
