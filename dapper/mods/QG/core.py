# "QG" Quasi-Geostrophic model.
# 
# Taken from Sakov's enkf-matlab package.
# Model is described in: 
# Sakov, Pavel, and Peter R. Oke.:
#   "A deterministic formulation of the ensemble Kalman filter:
#   an alternative to ensemble square root filters."
#   Tellus A 60.2 (2008): 361-371.
#
# More info:
#  - DAPPER/mods/QG/governing_eqn.png
#  - DAPPER/mods/QG/demo.py
#  - ψ (psi) is the stream function (i.e. surface elevation)
#  - Doubling time "between 25 and 50"
#  - Note Sakov's trick of increasing RKH2 from 2.0e-12 to 2.0e-11 to stabilize
#    the ensemble integration, which may be necessary for EnKF's with small N.
#    See example in mods/QG/cou09.py.

from dapper import *

#########################
# Model
#########################
try:
  from dapper.mods.QG.f90.py_mod import interface_mod as fortran
except ImportError as err:
  raise Exception("Have you compiled the (Fortran) model?" + \
      "\nSee README in folder "+dirs['dapper']+"/mods/QG/f90") from err

default_prms = OrderedDict([
  # These parameters may be interesting to change.
  ["dtout"        , 5.0        ], # dt for output to DAPPER.
  ["dt"           , 1.25       ], # dt used internally by Fortran. CFL = 2.0
  ["RKB"          , 0          ], # bottom     friction
  ["RKH"          , 0          ], # horizontal friction
  ["RKH2"         , 2.0e-12    ], # horizontal friction, biharmonic
  ["F"            , 1600       ], # Froud number
  ["R"            , 1.0e-5     ], # ≈ Rossby number
  ["scheme"       , "'rk4'"    ], # One of (2ndorder, rk4, dp5)
  # Do not change the following:
  ["tend"         , 0          ], # Only used by standalone QG
  ["verbose"      , 0          ], # Turn off
  ["rstart"       , 0          ], # Restart: switch
  ["restartfname" , "''"       ], # Restart: read file
  ["outfname"     , "''"       ]  # Restart: write file
  ])

class model_config:
  """
  Helps to ensure consistency between prms file (that Fortran module reads)
  and Python calls to step(), for example for dt.
  """

  def __init__(self, name, prms, mp=True):
    "Use prms={} to get the default configuration."

    # Insert prms. Assert key is present in defaults.
    D = default_prms.copy()
    for key in prms:
      assert key in D
      D[key] = prms[key]

    # Fortran code does not adjust its dt to divide dtout.
    # Nor is it worth implementing -- just assert:
    assert D['dtout']%D['dt'] == 0, "Must be integer multiple"

    self.prms  = D
    self.mp    = mp
    self.name  = name
    self.fname = os.path.join(os.path.dirname(__file__),'f90',
        'prms_' + name + '.txt')

    # Create string
    text = ["  %s = %s"%(key.ljust(20),str(D[key])) for key in D]
    text = """! Parameter namelist ("%s") generated via Python
    &parameters\n"""%name + "\n".join(text) + """\n/\n"""

    # Write string to file
    with open(self.fname, 'w') as f: f.write(text)


  def step_1(self, x0, t, dt):
    """Step a single state vector."""
    assert self.prms["dtout"] == dt  # Coz fortran.step() reads dt (dtout) from prms file.
    assert isinstance(t,float)       # Coz Fortran is typed.
    assert np.isfinite(t)            # QG is autonomous, but Fortran doesn't like nan/inf.
    psi = py2f(x0.copy())            # Copy coz Fortran will modify in-place.
    fortran.step(t,psi,self.fname)   # Call Fortran model.
    x = f2py(psi)                    # Flattening
    return x

  def step(self, E, t, dt):
    """Vector and 2D-array (ens) input, with multiproc for ens case."""
    if E.ndim==1:
      return self.step_1(E,t,dt)
    if E.ndim==2:
      if self.mp: # PARALLELIZED:
        # Note: the relative overhead for parallelization decreases
        # as the ratio dtout/dt increases.
        # But the overhead is already negligible with a ratio of 4.
        E = np.array(multiproc_map(self.step_1, E, t=t, dt=dt))
      else: # NON-PARALLELIZED:
        for n,x in enumerate(E): E[n] = self.step_1(x,t,dt)
      return E


#########################
# Domain management
#########################
# Domain size "hardcoded" in f90/parameters.f90.

# "Physical" domain length -- copied from f90/parameters.f90.
# In my tests, only square domains generate any dynamics of interest.
NX1 = 2
NY1 = 2
# Resolution level -- copied MREFIN from parameters.f90
res = 7
# Grid lengths.
nx = NX1 * 2 ** (res - 1) + 1 # (axis=1)
ny = NY1 * 2 ** (res - 1) + 1 # (axis=0)
# Actually, the BCs are psi = nabla psi = nabla^2 psi = 0,
# => psi should always be zero on the boundries.
# => it'd be safer to rm boundries from the DA state vector,
#    yielding ndim(state)=(nx-2)*(ny-2), but this is not done here.

# Fortran model (e.g. f90/interface.f90) requires orientation: X[ix,iy].
shape = (nx, ny)
# Passing arrays to/from Fortran requries that flags['F_CONTIGUOUS']==True.
order = 'F'
def    py2f(x):   return x.reshape(            shape      ,order=order)
def    f2py(X):   return X.flatten(                        order=order)
# However, FOR PRINTING/PLOTTING PURPOSES, the y-axis should be vertical
# [imshow(mat) uses the same orientation as print(mat)].
def  square(x):   return x.reshape(            shape[::-1])
def ind2sub(ind): return np.unravel_index(ind, shape[::-1])


#########################
# Free run
#########################
def gen_sample(model,nSamples,SpinUp,Spacing):
  simulator = with_recursion(model.step,prog="Simulating")
  K         = SpinUp + nSamples*Spacing
  Nx        = np.prod(shape) # total state length
  sample    = simulator(zeros(Nx), K, 0.0, model.prms["dtout"])
  return sample[SpinUp::Spacing]

sample_filename = os.path.join(dirs['samples'],'QG_samples.npz')
if not os.path.isfile(sample_filename):
  print('Did not find sample file',sample_filename,
      'for experiment initialization. Generating...')
  sample = gen_sample(model_config("sample_generation",{}),400,700,10)
  np.savez(sample_filename,sample=sample)



#########################
# Liveplotting
#########################
cm   = mpl.colors.ListedColormap(0.85*mpl.cm.jet(arange(256)))
cntr = nx*int(ny/2) + int(0.5*nx)
import dapper.tools.liveplotting as LP
def LP_setup(jj=None): return [
    (11, 1, LP.spatial2d(square,ind2sub,jj, cm) ),
    (15, 0, LP.spectral_errors ),
    (14, 0, LP.sliding_marginals(dims=cntr+arange(4)) ),
    ]




