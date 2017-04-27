# Reproduce results from Sakov and Oke "DEnKF" paper from 2008.

# TODO: Set RKH2 = 2.0e-11 ?
# TODO: Do something about parameter management,
#       especially dt
# TODO: Sampling X0 ala Sakov ?
# TODO: improve random_offset():
#        - no LCG, but better seed management
#        - random initialization


from common import *

from mods.QG.core import step, dt, nx, ny, m, S, square

# Considering that I have 8GB on the Mac, and the estimate:
# ≈ (8 bytes/float)*(129² float/stat)*(7 stat/k) * K,
# it should be possible to have an experiment with K≈8000.

#t = Chronology(dt=dt,dkObs=1,T=1200,BurnIn=100)
#t = Chronology(dt=dt,dkObs=1,T=600,BurnIn=100)
t = Chronology(dt=dt,dkObs=1,T=100,BurnIn=20)
#t = Chronology(dt=dt,dkObs=1,T=20,BurnIn=10)



mu0 = S[:,int(S.shape[1]*rand())]
# U should be scaled by svals?
U0 = np.load('mods/QG/svd_U.npz')['U']
X0 = GaussRV(mu=mu0,C=CovMat(10*U0,'Left'))

def show(x):
  im = plt.imshow(square(x))
  setter = lambda x:  im.set_data(square(x))
  return setter

f = {
    'm'    : m,
    'model': step,
    'noise': 0,
    'plot' : show,
    }



p  = 300
jj = equi_spaced_integers(m,p+1)[:p]

def random_offset(t):
  max_offset = jj[1]-jj[0]
  LCG(100*t)                # seed
  for _ in range(10): LCG() # burn in
  r = LCG()                 # draw
  return int(floor(max_offset * r))

def obs_inds(t):
  return jj + random_offset(t)

@ens_compatible
def hmod(E,t):
  return E[obs_inds(t)]

# Not validated
#def hjac(x,t):
#  H  = zeros((p,m))
#  for i,j in enumerate(obs_inds(t)):
#    H[i,j] = 1.0
#  return H

from aux.localization import inds_and_coeffs, unravel
xIJ = unravel(arange(m), (ny,nx)) # 2-by-m
def locf(radius,direction,t):
  """Prepare function:
    inds, coeffs = locf_at(state_or_obs_index)"""
  yIJ = xIJ[:,obs_inds(t)] # 2-by-p
  def ic(cntr, domain):
    return inds_and_coeffs(cntr, domain, (ny,nx), radius)
  if direction is 'x2y':
    # Could pre-compute ic() for all xIJ,
    # but speed gain is not worth it.
    def locf_at(ind):
      return ic(xIJ[:,ind], yIJ)
  elif direction is 'y2x':
    def locf_at(ind):
      return ic(yIJ[:,ind], xIJ)
  else: raise KeyError
  return locf_at

h = {
    'm'    : p,
    'model': hmod,
    'noise': GaussRV(C=4*eye(p)),
    'loc_f': locf,
    }

setup = OSSE(f,h,t,X0)
setup.name = os.path.relpath(__file__,'mods/')



####################
# Suggested tuning
####################



