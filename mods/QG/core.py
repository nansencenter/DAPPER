# "QG" Quasi-Geostrophic model.
# 
# Taken from Sakov's enkf-matlab package. See 
# Sakov, Pavel, and Peter R. Oke.:
#   "A deterministic formulation of the ensemble Kalman filter:
#   an alternative to ensemble square root filters."
#   Tellus A 60.2 (2008): 361-371.
#
#  - ψ is the stream function (i.e. surface elevation)
#  - Doubling time "between 25 and 50"
#
# See DAPPER/mods/QG/governing_eqn.png


import numpy as np

#
prm_filename = './mods/QG/f90/prms_tmp.txt'

#
dt_internal = 1.25 # CFL ≈ 2.0
dt          = 4*dt_internal 
dt_ratio    = dt/dt_internal
# The overhead for multiprocessing decreases for dt_ratio increases.
# But not really beyond dt_ratio = 4.
assert 0==dt_ratio%1.0, "Must be integer multiple"

# These parameters may be interesting to change. 
# In particular, RKH2=2.0e-11 yields a more stable integration.
prms = [
    ["dtout"        , dt         ], # dt registered by DAPPER
    ["dt"           , dt_internal], # dt used internally by Fortran
    ["RKB"          , 0          ], # bottom friction
    ["RKH"          , 0          ], # horizontal friction
    ["RKH2"         , 2.0e-12    ], # biharmonic horizontal friction
    ["F"            , 1600       ], # Froud number
    ["R"            , 1.0e-5     ], # ≈ Rossby number
    ["scheme"       , "'rk4'"    ]  # One of (2ndorder, rk4, dp5)
    ]
# Do NOT change:
prms2 = [
    ["tend"         , 0   ], # Only used by standalone QG
    ["verbose"      , 0   ], # Turn off
    ["rstart"       , 0   ], # Restart switch
    ["restartfname" , "''"], # Read from
    ["outfname"     , "''"]  # Write to
    ]
prms += prms2

# Create string
string = """! Parameter file auto generated from python
&parameters"""
for p in prms:
  string += "\n  " + p[0].ljust(20) + '= ' + str(p[1])
string += """\n/\n"""

# Write
with open(prm_filename, 'w') as f:
  f.write(string)




import scipy.io
S    = scipy.io.loadmat('mods/QG/f90/QG_samples-12.mat')['S']
m, N = S.shape
nx   = int(np.sqrt(m))
ny   = nx

def square(x):
  psi = x.copy()
  psi = psi.reshape((ny,nx),order='F')
  return psi

def flatten(psi):
  x = psi.ravel(order='F')
  return x

psi0 = square(S[:,-1])

from mods.QG.f90.py_mod import interface_mod as fortran
def step_1(x0, t, dt_):
  """Step a single state vector. Takes care of the copying"""
  assert dt_ == dt
  assert np.isfinite(t)
  assert isinstance(t,float)
  psi = square(x0)
  t   = np.array([t]) # QG is time-indep -- does not matter
  fortran.step(t,psi,prm_filename)
  x   = flatten(psi)
  return x


from aux.utils import multiproc_map
def step(E0, t, dt_):
  """Vector and 2D-array (ens) input, with multiproc for ens case."""
  if E0.ndim==1:
    return step_1(E0,t,dt_)
  if E0.ndim==2:
    return np.array(multiproc_map(step_1, E0, t=t,dt_=dt_))
    # Non-parallelized:
    #for n,x in enumerate(E0): #E[n] = step_1(x,t,dt_)


