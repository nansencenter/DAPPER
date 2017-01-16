#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""beta plane barotropic vorticity model.

This script uses a pseudospectral method to solve the barotropic vorticity
equation in two dimensions

    D/Dt[ω] = 0                                                             (1)

where ω = ξ + f.  ξ is local vorticity ∇ × u and f is global rotation.

Assuming an incompressible two-dimensional flow u = (u, v),
the streamfunction ψ = ∇ × (ψ êz) can be used to give (u,v)

    u = ∂/∂y[ψ]         v = -∂/∂x[ψ]                                        (2)

and therefore local vorticity can be given as a Poisson equation

    ξ = ∆ψ                                                                  (3)

where ∆ is the laplacian operator.  Since ∂/∂t[f] = 0 equation (1) can be
written in terms of the local vorticity

        D/Dt[ξ] + u·∇f = 0
    =>  D/Dt[ξ] = -vβ                                                       (4)

using the beta-plane approximation f = f0 + βy.  This can be written entirely
in terms of the streamfunction and this is the form that will be solved
numerically.

    D/Dt[∆ψ] = +β ∂/∂x[ψ]                                                   (5)

The spectral method defines ψ as a Fourier sum

    ψ = Σ A(t) exp(i (kx + ly))

and as such spatial derivatives can be calculated analytically

    ∂/∂x[ψ] = ikψ       ∂/∂y[ψ] = ilψ

The pseudospectral method will use the analytic derivatives to calculate
values for (u, v) which will then be used to evaluate nonlinear terms.


References:
* This code was developed based on a MATLAB script bvebb.m
  (Original source Dr. James Kent & Prof. John Thuburn)
* And the GFDL documentation for spectral barotropic models
  found here 
  [http://www.gfdl.noaa.gov/cms-filesystem-action/user_files/pjp/barotropic.pdf]
* McWilliams Initial Condition inspired by pyqg [https://github.com/pyqg/pyqg]


Downloaded from github.com/jamesp/shallowwater
Modified for DAPPER by Patrick N. Raanes
"""


import matplotlib.pyplot as plt
import numpy as np

from numpy import pi, cos, sin
from numpy.fft import fftshift, fftfreq
from numpy.fft.fftpack import rfft2, irfft2

from aux.utils import Timer
from aux.misc import validate_int



### Configuration
nx = 256
ny = 256                        # numerical resolution
Lx = 1.0
Ly = 1.0                        # domain size [m]
ubar = 0.00                     # background zonal velocity  [m/s]
beta = 8.0                      # beta-plane f = f0 + βy     [1/s 1/m]
n_diss = 2.0                    # Small-scale dissipation of the form ∆^2n_diss,
                                # such that n_diss = 2 would be a ∆^4 hyperviscosity. 
tau = 0.3                       # coefficient of dissipation
                                # (smaller => more dissipation)

#Poorly determined coefficients for forcing and dissipation
#r_rayleigh = (1./50000.)/np.sqrt(10.)
r_rayleigh = (1./500.)/np.sqrt(10.)
forcing_amp_factor=100.0/np.sqrt(1.)


### Function Definitions
def ft(phi):
    """Go from physical space to spectral space."""
    return rfft2(phi, axes=(-2, -1))

def ift(psi):
    """Go from spectral space to physical space."""
    return irfft2(psi, axes=(-2,-1))

def courant_number(psix, psiy, dx, dt):
    """Calculate the Courant Number given the velocity field and step size."""
    maxu = np.max(np.abs(psiy))
    maxv = np.max(np.abs(psix))
    maxvel = maxu + maxv
    return maxvel*dt/dx

def grad(phit):
    """Returns the spatial derivatives of a Fourier-transformed variable.
    Returns (∂/∂x[F[φ]], ∂/∂y[F[φ]]) i.e. (ik F[φ], il F[φ])"""
    global ik, il
    phixt = ik*phit        # d/dx F[φ] = ik F[φ]
    phiyt = il*phit        # d/dy F[φ] = il F[φ]
    return (phixt, phiyt)

def velocity(psit):
    """Returns the velocity field (u, v) from F[ψ]."""
    psixt, psiyt = grad(psit)
    psix = ift(psixt)    # v =   ∂/∂x[ψ]
    psiy = ift(psiyt)    # u = - ∂/∂y[ψ]
    return (-psiy, psix)

def spectral_variance(phit):
    global nx, ny
    var_density = 2.0 * np.abs(phit)**2 / (nx*ny)
    var_density[:,0] /= 2
    var_density[:,-1] /= 2
    return var_density.sum()
    
def anti_alias(phit):
    """Set the coefficients of wavenumbers > k_mask to be zero."""
    k_mask = (8./9.)*(nk+1)**2.
    phit[(np.abs(ksq/(dk*dk)) >= k_mask)] = 0.0    
    
def high_wn_filter(phit):
    """Applies the high wavenumber filter of smith et al 2002"""
    filter_dec = -np.log(1.+2.*pi/nk)/((nk-kcut)**filter_exp)
    filter_idx = np.abs(ksq/(dk*dk)) >= kcut**2.
    phit[filter_idx] *= exp(filter_dec*(np.sqrt(ksq[filter_idx]/(dk*dk))-kcut)**filter_exp)
    
def forcing_spatial_mask(phi):    
   """TODO: Make spatial mask for forcing"""
   phi[idx_sp_mask] *= 0.


_prhs, _pprhs  = 0.0, 0.0  # previous two right hand sides
def adams_bashforth(zt, rhs, dt):
    """Take a single step forward in time using Adams-Bashforth 3."""
    global step, _prhs, _pprhs
    #print("step:",step)
    if step is 0:
        # forward euler
        dt1 = dt
        dt2 = 0.0
        dt3 = 0.0
    elif step is 1:
        # AB2 at step 2
        dt1 = 1.5*dt
        dt2 = -0.5*dt
        dt3 = 0.0
    else:
        # AB3 from step 3 on
        dt1 = 23./12.*dt
        dt2 = -16./12.*dt
        dt3 = 5./12.*dt

    newzt = zt + dt1*rhs + dt2*_prhs + dt3*_pprhs  
    _pprhs = _prhs
    _prhs  = rhs
    return newzt



## SETUP

### Physical Domain
dx = Lx / nx
dy = Ly / ny
#dt = 0.4 * 16.0 / nx          # choose an initial dt. This will change
                              # as the simulation progresses to maintain
                              # numerical stability
# Time params
t            = 0.0
tmax         = 10000
dt           = 0.1
PLOT_EVERY_S = 0.2
tplot        = t + PLOT_EVERY_S
FIRST        = True



# Make meshgrid of y coordinates
y = np.linspace(0, Ly, num=ny)
y_arr = np.flipud(np.tile(y,(nx,1)).transpose())


### Spectral Domain
nl = validate_int(ny)
nk = validate_int(nx/2 + 1)
dk = 2.0*pi/Lx
dl = 2.0*pi/Ly
# calculate the wavenumbers [1/m]
# The real FT has half the number of wavenumbers in one direction:
# FT_x[real]    -> complex : 1/2 as many complex numbers needed as real signal
# FT_y[complex] -> complex : After the first transform has been done the signal
# is complex, therefore the transformed domain in second dimension is same size
# as it is in euclidean space.
# Therefore FT[(nx, ny)] -> (nx/2, ny)
# The 2D Inverse transform returns a real-only domain (nx, ny)
k = dk*np.arange(0, nk)     [np.newaxis, :]
l = dl*fftfreq(nl, d=1.0/nl)[:, np.newaxis]

ksq = k**2 + l**2               # squared wavenums (yields laplacian)
ksq[ksq == 0] = 1.0             # avoid divide by zero - set ksq = 1 at zero wavenum
rksq = 1.0 / ksq                # reciprocal 1/(k^2 + l^2)

ik = 1j*k                       # wavenumber mul. imaginary unit is useful
il = 1j*l                       # for calculating derivatives

## Dissipation & Spectral Filters
# Use ∆^2n_diss hyperviscosity to diffuse at small scales
# (i.e. n_diss = 2 would be ∆^4)
# use the x-dimension for reference scale values
nu = ((Lx/(np.floor(nx/3)*2.0*pi))**(2*n_diss))/tau

# High wavenumber filter coefficients.
# Any waves with wavenumber below kcut are not dissipated at all.
filter_exp = 8.
kcut = 30.

# Spectral Filter as per [Arbic and Flierl, 2003]
wvx = np.sqrt((k*dx)**2 + (l*dy)**2)
spectral_filter = exp(-23.6*(wvx-0.65*pi)**4)
spectral_filter[wvx <= 0.65*pi] = 1.0


# ζ and F[ζ] arrays
z  = np.zeros((ny, nx), dtype=np.float64)
zt = np.zeros((nl, nk), dtype=np.complex128)

### Diagnostic arrays
time_arr       = np.zeros(1)
tot_energy_arr = np.zeros(1)


amp = 636

def integrator(xt,t0,dt_tot):

  t = t0

  nsteps = validate_int(dt_tot/dt)
  if nsteps < 10:
    raise Warning('dt_tot not long enough to properly bootstrap\
    multiSTEP (adams-bashford) integrator\
    (and relying solely on Euler is too unstable).')

  global step, _prhs, _pprhs
  step = _prhs = _pprhs = 0

  for k in range(nsteps):
    # calculate derivatives in spectral space
    psit         = -rksq * xt # F[ψ] = - F[ζ] / (k^2 + l^2)
    psixt, psiyt = grad(psit)
    zxt, zyt     = grad(xt)

    # transform back to physical space for pseudospectral part
    z[:] = ift(xt)
    psix = ift(psixt)
    psiy = ift(psiyt)
    zx   = ift(zxt)
    zy   = ift(zyt)

    # Non-linear: calculate the Jacobian in real space
    # and then transform back to spectral space
    jac  = psix * zy - psiy * zx + ubar * zx
    jact = ft(jac)

    # apply forcing in spectral space by exciting certain wavenumbers
    # (could also apply some real-space forcing and then convert
    # into spectral space before adding to rhs) 
    rstream     = np.random.RandomState(int(t*10000))
    forcet      = np.zeros_like(ksq, dtype=complex)
    idx         = (14 < np.sqrt(ksq)/dk) & (np.sqrt(ksq)/dk < 20)
    numr        = ksq.shape[0]*ksq.shape[1]
    r_amp       = rstream.rand(numr).reshape(ksq.shape)[idx]
    r_phs       = rstream.rand(numr).reshape(ksq.shape)[idx]
    forcet[idx] = 0.5*amp*(r_amp - 0.5)*exp(1j*2.*pi*r_phs)

    c = 999

    # take a timestep and diffuse
    rhs   = -jact + beta*psixt - xt*r_rayleigh + forcet 
    xt[:] = adams_bashforth(xt, rhs, dt)
    
    #anti_alias
    anti_alias(xt)

    #high_wavenumber_filter
    high_wn_filter(xt)

    t    += dt
    step += 1

    global tplot, time_arr, tot_energy_arr, FIRST
    if t > tplot and __name__ == "__main__":
        with Timer('Diagnostics'):
          urms=np.sqrt(np.mean(psix**2 + psiy**2))
          rhines_scale = np.sqrt(urms/beta)
          tot_energy=0.5*urms**2.
          epsilon = 2 * r_rayleigh * tot_energy
          l_epsilon = (epsilon / beta**3.)**(1./5.)
          
          time_arr=np.append(time_arr,t)
          tot_energy_arr=np.append(tot_energy_arr,tot_energy)
          
          force=ift(forcet)

        if FIRST:
          #plt.ion() # plot in realtime
          plt.figure(figsize=(15, 12))
          #plt.clf()
          FIRST = False

          with Timer('vorticity (FIRST)'):
            axv = plt.subplot2grid((2,3), (0, 0), colspan=2, rowspan=2)
            vh = plt.imshow(z, extent=[0, Lx, 0, Ly], cmap=plt.cm.YlGnBu)
            plt.xlabel('x')
            plt.ylabel('y')
            zmax = np.max(np.abs(z))
            plt.clim(-zmax,zmax)
            plt.colorbar(orientation='horizontal')
            plt.title('Rel. vorticity at {:.2f}s dt={:.2f}'.format(t, dt)) 

          with Timer('Forcing'):
            plt.subplot2grid((2,3), (0, 2))
            fh = plt.imshow(force, extent=[0, Lx, 0, Ly], cmap=plt.cm.YlGnBu)
            plt.xlabel('x')
            plt.ylabel('y')
            forcemax = np.max(np.abs(force))
            plt.clim(-forcemax,forcemax)
            plt.colorbar(orientation='horizontal')
            plt.title('Forcing')      

          with Timer('Energy'):
            axl = plt.subplot2grid((2,3), (1, 2))
            lh = plt.plot(time_arr, tot_energy_arr)[0]
            plt.xlabel('Time')
            plt.ylabel('Total Energy')

            plt.subplots_adjust(wspace=0.4,left=0.05,right=0.97)
        else:
          with Timer('Update'):
            vh.set_data(z)
            axv.set_title('Rel. vorticity at {:.2f}s dt={:.2f}'.format(t, dt)) 
            fh.set_data(force)
            lh.set_data(time_arr, tot_energy_arr)
            axl.set_xlim(right=t)
            #axl.set_ylim((np.min(tot_energy_arr), np.max(tot_energy_arr)))
            axl.set_ylim((0.0000043, 0.00000435))
            axl.ticklabel_format(style='sci',useOffset=4e-6,axis='y')

        plt.pause(0.01)
        print('[{:5d}] {:.2f} Max z: {:2.2f} c={:.2f} dt={:.2f} rh_s={:.3f} l_eps={:.3f} ratio={:2.2f} urms={:2.2f}'.format(
            step, t, np.max(z), c, dt, rhines_scale, l_epsilon, rhines_scale/l_epsilon, urms))
        tplot = t + PLOT_EVERY_S

  return
  

if __name__ == "__main__":
  z = np.load('mods/Barotropic/z0_1.npz')['z']
  z = z[:ny,:nx]
  zt[:] = ft (z)

  integrator(zt,t,tmax)



### Initial Condition
#   # The McWilliams Initial Condition from [McWilliams - J. Fluid Mech. (1984)]
#   ck   = np.zeros_like(ksq)
#   ck   = np.sqrt(ksq + (1.0 + (ksq/36.0)**2))**-1
#   piit = rand(*ksq.shape)*ck + 1j*rand(*ksq.shape)*ck
#   
#   pii  = ift(piit)
#   pii  = pii - pii.mean()
#   piit = ft(pii)
#   KE   = spectral_variance(piit*np.sqrt(ksq)*spectral_filter)
#   
#   qit  = -ksq * piit / np.sqrt(KE)
#   qi   = ift(qit)
#   z[:] = qi
#   
#   # calc a reasonable forcing amplitude
#   amp = forcing_amp_factor * np.max(np.abs(qi))
#   
#   # initialise the transformed ζ
#   zt[:] = ft (z)
#   anti_alias (zt)
#   z[:]  = ift(zt)
#   
#   # initialise the storage arrays
#   time_arr[0]=t
#   
#   psit = -rksq * zt         # F[ψ] = - F[ζ] / (k^2 + l^2)
#   psixt, psiyt = grad(psit)
#   psix = ift(psixt)
#   psiy = ift(psiyt)
#       
#   urms              = np.sqrt(np.mean(psix**2 + psiy**2))
#   tot_energy        = 0.5*urms**2.
#   tot_energy_arr[0] = tot_energy

