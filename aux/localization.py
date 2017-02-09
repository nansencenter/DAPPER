from common import *

# Defaults
CUTOFF   = 1e-3
TAG      = 'GC'

def unravel(inds, shape, order='C'):
  """
  Compute (i,j) subscripts of
  the ravelled (i.e. vectorized) <inds>
  of a <shape> matrix.

  Note that unravelling is slow, and should only be done once.
  By contrast, the dist/coeff computations are faster,
  but could require much memory to pre-compute
  and so should be computed on the fly.
  """
  inds  = np.atleast_1d(inds)
  shape = np.atleast_1d(shape)

  IJ = asarray(np.unravel_index(inds, shape, order=order))

  for i,dim in enumerate(IJ): assert max(dim) < shape[i]
  return IJ


# TODO: Can replace by from scipy.spatial.distance ?
def distance_nD(centr, domain, shape, periodic=True):
  """
  Euclidian distance between centr and domain,
  both of which are indices of x.ravel(order='C'),
  where x.shape==<shape>.
  Vectorized for multiple domain pts.
  """
  shape = np.atleast_1d(shape)

  cIJ = centr.reshape((-1,1))
  dIJ = domain

  delta = abs(cIJ - dIJ)
  if periodic:
    shape = shape[:,np.newaxis]
    delta = numpy.where(delta>shape/2, shape-delta, delta)

  return sla.norm(delta,axis=0)


def dist2coeff(dists, radius, tag=TAG):
  """Compute coefficients corresponding to a distances."""
  coeffs = zeros(dists.shape)

  if tag == 'Gauss':
    R = radius
    coeffs = exp(-0.5 * (dists/R)**2)
  elif tag == 'Exp':
    R = radius
    coeffs = exp(-0.5 * (dists/R)**3)
  elif tag == 'Cubic':
    R            = radius * 1.8676
    inds         = dists <= R
    coeffs[inds] = (1 - (dists[inds] / R) ** 3) ** 3
  elif tag == 'Quadro':
    R            = radius * 1.7080
    inds         = dists <= R
    coeffs[inds] = (1 - (dists[inds] / R) ** 4) ** 4
  elif tag == 'GC':
    # Gaspari_Cohn
    R = radius * 1.7386
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

  return coeffs


def inds_and_coeffs(centr, domain, domain_shape,
    radius, cutoff=CUTOFF, tag=TAG):
  """
  Returns:
  inds   = the **indices of** domain that "close to" centr,
           such that the local domain is: domain[inds].
  coeffs = the corresponding coefficients.
  """
  dists  = distance_nD(centr, domain, domain_shape)

  coeffs = dist2coeff(dists, radius, tag)

  # Truncate with cut-off
  inds   = arange(len(dists))[coeffs > cutoff]
  coeffs = coeffs[inds]

  return inds, coeffs


