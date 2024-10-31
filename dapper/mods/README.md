See the README section on
[test cases (models)](../../index.md#test-cases-models)
for a table overview of the included models.

## Defining your own model

Below is a sugested structuring followed by most models already within DAPPER.
However, you are free to organize your model as you see fit,
as long as it culminates in the definition of one or more [`mods.HiddenMarkovModel`][].
For the sake of modularity,
try not to import stuff from DAPPER outside of [`mods`](.) and [`tools.liveplotting`][].

- Make a directory: `my_model`. It does not have to reside within the `dapper/mods` folder,
  but make sure to look into some of the other dirs thereunder as examples,
  for example `dapper/mods/DoublePendulum`.
- Make a file: `my_model/__init__.py` to hold the core workings of the model.
  Further details are given [below](#details-on-my_model__init__py), but the
  main work lies in defining a `step(x, t, dt)` function
  (you can name it however you like, but `step` is the convention),
  to implement the dynamical model/system mapping the state `x`
  from one time `t` to another `t + dt`.
- Make a file: `my_model/demo.py` to run `step` and visually showcase
  a simulation of the model without any DA, and verify it's working.
- Make a file: `my_model/my_settings_1.py` that defines
  (or "configures", since there is usually little programming logic and flow taking place)
  a complete [`mods.HiddenMarkovModel`][] ready for a synthetic experiment
  (also called "twin experiment" or OSSE).
- Once you've made some experiments you believe are noteworthy you should add a
  "suggested settings/tunings" section in comments at the bottom of
  `my_model/my_settings_1.py`, listing some of the relevant DA method
  configurations that you tested, along with the RMSE (or other stats) that
  you obtained for those methods.  You will find plenty of examples already in
  DAPPER, used for cross-referenced with literature to verify the workings of DAPPER
  (and the reproducibility of publications).


### Details on `my_model/__init__.py`

- The `step` function must support 2D-array (i.e. ensemble)
  and 1D-array (single realization) input, and return output of the same
  number of dimensions (as the input).
  See

    - [`mods.Lorenz63`][]: use of `ens_compatible`.
    - [`mods.Lorenz96`][]: use of relatively clever slice notation.
    - [`mods.LorenzUV`][]: use of cleverer slice notation: `...` (ellipsis).
      Consider pre-defining the slices like so:
      ```python
      iiX = (..., slice(None, Nx))
      iiP = (..., slice(Nx, None))
      ```
      to abbreviate the indexing elsewhere.

    - [`mods.QG`][]: use of parallelized for loop (map).

    !!! note
        To begin with, test whether the model works on 1 realization,
        before running it with several (simultaneously).
        Also, start with a small integration time step,
        before using more efficient/adventurous time steps.
        Note that the time step might need to be shorter in assimilation,
        because it may cause instabilities.

    !!! note
        Most models are defined using simple procedural style.
        However, [`mods.LorenzUV`][] and [`mods.QG`][] use OOP,
        which is perhaps more robust when different
        control-variable settings are to be investigated.
        The choice is yours.

        In parameter estimation problems, the parameters are treated as input
        variables to the "forward model". This does not *necessarily* require OOP.
        See `docs/examples/param_estim.py`.

- Optional: define a suggested/example initial state, `x0`.
  This facilitates the specification of initial conditions for different synthetic
  experiments, as random variables centred on `x0`.  It is also a
  convenient way just to specify the system size as `len(x0)`.  In many
  experiments, the specific value of `x0` does not matter, because most
  systems are chaotic, and the average of the stats are computed only for
  `time > BurnIn > 0`, which will not depend on `x0` if the experiment is
  long enough.  Nevertheless, it's often convenient to pre-define a point
  on the attractor, or basin, or at least ensure "physicality", for
  quicker spin-up (burn-in).

- Optional: define a number called `Tplot` which defines
  the (sliding) time window used by the liveplotting of diagnostics.

- Optional: To use the (extended) Kalman filter, or 4D-Var,
  you will need to define the model linearization, typically called `dstep_dx`.
  Note: this only needs to support 1D input (single realization).
