from common import *

from mods.Barotropic.baro_vort_simple import integrator, nl, nk, nx, ny, ft, ift, Lx, Ly

t = Chronology(1,dkObs=1,T=40,BurnIn=2)

z0  = np.load('mods/Barotropic/z0_1.npz')['z'][:ny,:nx]



# Forward model
# State: [zeta]
# Slowdown (due to wrapping FT/iFT) only about 7%
m   = nx*ny
mu0 = z0.ravel()
#
@atmost_2d
def step_ens(E, t, dt):
  E_next = E.copy()
  for x in E_next:
    xt = ft(x.reshape(ny,nx))

    integrator(xt,t,dt)  

    z    = ift(xt)
    x[:] = z.ravel()

    #plt.figure(2).clf()
    #plt.imshow(z, extent=[0, Lx, 0, Ly], cmap=plt.cm.YlGnBu,clim=(-0.9,0.9))
    #plt.pause(0.01)
  return E_next


f = {
    'm'    : m,
    'model': step_ens,
    'noise': 0
    }

UsT = np.load('mods/Barotropic/C12.npz')['xx']
C12 = UsT/sqrt(UsT.shape[0] - 1)
X0  = GaussRV(mu0, CovMat(Right = 0.03*C12))
#X0 = RV(func= lambda N: mu0 + 1e-3*randn((N,len(mu0))))

p = 300
obsInds = equi_spaced_integers(m,p)
@atmost_2d
def hmod(E,t):
  return E[:,obsInds]

h = {
    'm'    : p,
    'model': hmod,
    'noise': GaussRV(C=0.1**2*eye(p)),
    #'noise': GaussRV(C=0.01**2*eye(p)),
    }

from aux.localization import inds_and_coeffs, unravel
def loc_wrapper(radius,direction=None):
  # Unravelling is slow, and should only be done once.
  # By contrast, the dist/coeff computations are not so slow,
  # and maybe it's better to compute on the go coz:
  #  - less memory
  #  - more flexibility to future changes.
  iix = arange(m)
  dIJ = unravel(iix    , (ny,nx))
  oIJ = unravel(obsInds, (ny,nx))
  if direction is 'x2y':
    inds   = np.empty(m,np.ndarray)
    coeffs = np.empty(m,np.ndarray)
    for i in range(m):
      inds[i], coeffs[i] = inds_and_coeffs(dIJ[:,i], oIJ, (ny,nx), radius)
    def locf(i):
      return inds[i], coeffs[i]
    #def locf(i):
      #return inds_and_coeffs(dIJ[:,i], oIJ, (ny,nx), radius)
      #return inds_and_coeffs(i, obsInds, (ny,nx), radius,do_unravel=True)
  elif direction is 'y2x':
    def locf(i):
      return inds_and_coeffs(oIJ[:,i], dIJ, (ny,nx), radius)
      #return inds_and_coeffs(obsInds[i], iix, (ny,nx), radius, do_unravel=True)
  else: raise KeyError
  return locf
 

setup = TwinSetup(f,h,t,X0)
setup.name = os.path.relpath(__file__,'mods/')
setup.locf = loc_wrapper
setup.nYnX = (ny, nx)


# Alternative forward model wrapper
# State: [real(FT(zeta)), imag(FT(zeta))]
#   m   = 2*nk*nl # real & imag => 2
#   mu0 = ft(z0).ravel()
#   mu0 = np.concatenate([np.real(mu0), np.imag(mu0)])
#   #
#   @atmost_2d
#   def step_ens(E, t, dt):
#     E_next = E.copy()
#     for x in E_next:
#       xt = (x[:m//2] + 1j*x[m//2:]).reshape(nl,nk)
#   
#       integrator(xt,t,dt)
#   
#       #z = ift(xt)
#       #plt.figure(2).clf()
#       #plt.imshow(z, extent=[0, Lx, 0, Ly], cmap=plt.cm.YlGnBu,clim=(-0.9,0.9))
#       #plt.pause(0.01)
#   
#       xt = xt.ravel()
#       x[:m//2] = np.real(xt)
#       x[m//2:] = np.imag(xt)
#     return E_next

