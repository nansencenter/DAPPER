from common import *

from mods.Barotropic.baro_vort_simple import integrator, nl, nk, nx, ny, ft, ift, Lx, Ly

t = Chronology(1,dkObs=1,T=40,BurnIn=2)

z0  = np.load('mods/Barotropic/z0_1.npz')['z'][:ny,:nx]


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


#X0  = GaussRV(mu0, 0.005**2*eye(len(mu0))) Must implement sparse GaussRV
X0 = RV(sampling_func = lambda N: mu0 + 1e-3*randn((N,len(mu0))))



p = 300
obsInds = equi_spaced_integers(m,p)
@atmost_2d
def hmod(E,t):
  return E[:,obsInds]
h = {
    'm'    : p,
    'model': hmod,
    'noise': GaussRV(C=0.001**2*eye(p)),
    #'noise': GaussRV(C=0.01**2*eye(p)),
    }
 
other = {'name': os.path.relpath(__file__,'mods/')}

setup = OSSE(f,h,t,X0,**other)
