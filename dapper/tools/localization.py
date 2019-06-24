from dapper import *

# A good introduction to localization:
# Sakov (2011), Computational Geosciences:
# "Relation between two common localisation methods for the EnKF".

# Defaults
CUTOFF   = 1e-3
TAG      = 'GC'


def distance_nd(centre, pts, shape, periodic=True):
  """Euclidian distance between centre and pts.

  - centre: 
  - pts: an ndim-by-npts array.
  - shape: tuple specifying the domain extent.
  """

  # Make col vectors, to enable (ensure) broadcasting...
  centre = np.reshape(centre,(-1,1))
  shape  = np.reshape(shape ,(-1,1))
  # ... in this subtraction
  d = abs(centre - pts)

  if periodic:
    d = np.minimum(d, shape-d)

  return sla.norm(d,axis=0)


def dist2coeff(dists, radius, tag=None):
  """Compute tapering coefficients corresponding to a distances.

  NB: The radius is internally adjusted such that, independently of 'tag',
  coeff==exp(-0.5) when distance==radius.

  This is largely based on Sakov's enkf-matlab code. Two bugs have here been fixed:
   - The constants were slightly wrong, as noted in comments below.
   - It forgot to take sqrt() of coeffs when applying them through 'local analysis'.
  """
  coeffs = zeros(dists.shape)

  if tag is None:
    tag = TAG

  if tag == 'Gauss':
    R = radius
    coeffs = exp(-0.5 * (dists/R)**2)
  elif tag == 'Exp':
    R = radius
    coeffs = exp(-0.5 * (dists/R)**3)
  elif tag == 'Cubic':
    R            = radius * 1.87 # Sakov: 1.8676
    inds         = dists <= R
    coeffs[inds] = (1 - (dists[inds] / R) ** 3) ** 3
  elif tag == 'Quadro':
    R            = radius * 1.64 # Sakov: 1.7080
    inds         = dists <= R
    coeffs[inds] = (1 - (dists[inds] / R) ** 4) ** 4
  elif tag == 'GC':
    # Gaspari_Cohn
    R = radius * 1.82 # Sakov: 1.7386
    #
    ind1         = dists<=R
    r2           = (dists[ind1] / R) ** 2
    r3           = (dists[ind1] / R) ** 3
    coeffs[ind1] = 1 + r2 * (- r3 / 4 + r2 / 2) + r3 * (5 / 8) - r2 * (5 / 3)
    #
    ind2         = np.logical_and(R < dists, dists <= 2*R)
    r1           = (dists[ind2] / R)
    r2           = (dists[ind2] / R) ** 2
    r3           = (dists[ind2] / R) ** 3
    coeffs[ind2] = r2 * (r3 / 12 - r2 / 2) + r3 * (5 / 8) + r2 * (5 / 3) - r1 * 5 + 4 - (2 / 3) / r1
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
  inds   = arange(len(dists))[coeffs > cutoff]
  coeffs = coeffs[inds]

  return inds, coeffs



def rectangular_partitioning(shape,steps):
  """N-D rectangular batch generation.

  shape: [len(grid[dim]) for dim in range(ndim)]
  steps: [step_len[dim]  for dim in range(ndim)]

  returns a list of batches,
  where each element (batch) is an ndim-list,
  where each element is an array of ordinates (length == #gridpoints in batch).

  # Example, with visualization:
  >>> shape   = [4,13]
  >>> steps   = [2,4]
  >>> batches = rectangular_partitioning(shape, steps)
  >>> M       = np.prod(shape)
  >>> nB      = len(batches)
  >>> values  = np.random.choice(arange(nB),nB,0)
  >>> Z       = zeros(shape)
  >>> for ib,b in enumerate(batches):
  >>>   Z[b] = values[ib]
  >>> plt.imshow(Z)
  """
  assert len(shape)==len(steps)
  ndim = len(steps)

  # An ndim list of (average) local grid lengths:
  nLocs = [round(n/d) for n,d in zip(shape,steps)]
  # An ndim list of (marginal) grid partitions [array_split() handles non-divisibility]:
  edge_partitions = [np.array_split(np.arange(n),nLoc) for n,nLoc in zip(shape,nLocs)]

  batches = []
  for batch_edges in itertools.product(*edge_partitions):
    # The 'indexing' argument below is actually inconsequential:
    # it merely changes batch's internal ordering.
    batch_rect    = np.meshgrid(*batch_edges, indexing='ij')
    coords        = [ ii.flatten() for ii in batch_rect]
    batches      += [ coords ]

  return batches


# NB: Don't try to put the time-dependence of obs_inds inside obs_taperer().
# That would require calling ind2sub len(batches) times per analysis,
# and the result cannot be easily cached, because of multiprocessing.
def obs_inds_safe(obs_inds, t):
  "Support time-dependent obs_inds."
  try:              return obs_inds(t)
  except TypeError: return obs_inds


# NB: Why is the 'order' argument not supported by this module? Because:
#  1) Assuming only order (orientation) 'C' simplifies the module's code.
#  2) It's not necessary, because the module only communicates to *exterior* via indices
#     [of what assumes to be X.flatten(order='C')], and not coordinates!
#     Thus, the only adaptation necessary if the order is 'F' is to reverse
#     the shape parameter passed to these functions (example: mods/QG/sak08).


def partial_direct_obs_nd_loc_setup(shape,batch_shape,obs_inds,periodic):
  "N-D rectangle"

  def sub2ind(sub): return np.ravel_multi_index(sub, shape)
  def ind2sub(ind): return np.    unravel_index(ind, shape)


  M = np.prod(shape)
  state_coord = ind2sub(arange(M))

  batches = rectangular_partitioning(shape, batch_shape)
  batches = [sub2ind(b) for b in batches]

  def loc_setup(radius,direction,t,tag=None):
    "Provide localization setup for time t."
    obs_inds_now = obs_inds_safe(obs_inds,t)
    obs_coord = ind2sub( obs_inds_now )

    if direction is 'x2y':
      def obs_taperer(batch):
        # Don't use "batch = batches[iBatch]" (with iBatch as this function's input).
        # This would slow down multiproc., coz batches gets copied to each process.
        batch_center_coord = array(ind2sub(batch)).mean(axis=1)
        dists = distance_nd(batch_center_coord, obs_coord, shape, periodic)
        return inds_and_coeffs(dists, radius, tag=tag)
      return batches, obs_taperer

    elif direction is 'y2x':
      def state_taperer(iObs):
        obs_j_coord = ind2sub(obs_inds_now[iObs])
        dists = distance_nd(obs_j_coord, state_coord, shape, periodic)
        return inds_and_coeffs(dists, radius, tag=tag)
    return state_taperer

  return loc_setup



def no_localization(Nx,Ny):

  def obs_taperer(batch ): return arange(Ny), ones(Ny)
  def state_taperer(iObs): return arange(Nx), ones(Nx)

  def loc_setup(radius,direction,t,tag=None):
    """Returns all indices, with all tapering coeffs=1.

    Useful for testing local DA methods without localization
    e.g. if LETKF <==> EnKF('Sqrt')."""
    assert radius == np.inf, "Localization functions not specified"

    if   direction is 'x2y': return [arange(Nx)], obs_taperer
    elif direction is 'y2x': return             state_taperer

  return loc_setup




