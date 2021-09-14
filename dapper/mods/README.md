See the README section on
[test cases (models)](https://github.com/nansencenter/DAPPER#Test-cases-models)
for an overview of the models included with DAPPER.

The models are all simple;
this facilitates the reliability, reproducibility, and attributability
of the experiment results.

## Defining your own model

Follow the example of one of the models within the `dapper/mods` folder.
Essentially, you just need to define all of the attributes of a
`dapper.mods.HiddenMarkovModel`.
To make sure this is working, we suggest the following structure:

- Make a directory: `my_model`

- Make a file: `my_model/__init__.py` where you define the core
  workings of the model.
    - Typically, this culminates in a `step(x, t, dt)` function,
      which defines the dynamical model/system mapping the state `x`
      from one time `t` to another `t + dt`.
      This model "operator" must support 2D-array (i.e. ensemble)
      and 1D-array (single realization) input, and return output of the same
      number of dimensions (as the input).
      See

        - `dapper.mods.Lorenz63`: use of `ens_compatible`.
        - `dapper.mods.Lorenz96`: use of relatively clever slice notation.
        - `dapper.mods.LorenzUV`: use of cleverer slice notation: `...` (ellipsis).
          Consider pre-defining the slices like so:

                iiX = (..., slice(None, Nx))
                iiP = (..., slice(Nx, None))

            to abbreviate the indexing elsewhere.

        - `dapper.mods.QG`: use of parallelized for loop (map).

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

- Most models are defined using a procedural and function-based style.
  However, `dapper.mods.LorenzUV` and `dapper.mods.QG` use OOP.
  This is more flexible & robust, and better suited when different
  control-variable settings are to be investigated.

    .. note::
        In parameter estimation problems, the parameters are treated as input
        variables to the "forward model". This does not necessarily require
        OOP. See `examples/param_estim.py`.

- Make a file: `my_model/demo.py` to visually showcase
  a simulation of the model, and verify it's working.

    .. hint::
        To begin with, test whether the model works on 1 realization,
        before running it with several (simultaneously).
        Also, start with a small integration time step,
        before using more efficient/adventurous time steps.
        Note that the time step might need to be shorter in assimilation,
        because it may cause instabilities.

- Ideally, both `my_model/__init__.py` and `my_model/demo.py`
  do not rely on components of DAPPER outside of `dapper.mods`.

- Make a file: `my_model/my_settings_1.py` that defines
    (or "configures", since there is usually little actual programming taking place)
    a complete Hidden Markov Model ready for a synthetic experiment
    (also called "twin experiment" or OSSE).
    See `dapper.mods.HiddenMarkovModel` for details on what this requires.
    Each existing model comes with several examples of model settings from the literature.
    See, for example, `dapper.mods.Lorenz63.sakov2012`.

    .. warning::
      These configurations do not necessarily hold a very high programming standard,
      as they may have been whipped up at short notice to replicate some experiments,
      and are not intended for re-use.

      Nevertheless, sometimes they are re-used by another configuration script,
      leading to a major gotcha/pitfall: changes made to the imported `HMM` (or
      the model's module itself) also impact the original object (since they
      are mutable and thereby referenced).  This *usually* isn't an issue, since
      one rarely imports two/more separate configurations. However, the test suite
      imports all configurations, which might then unintentionally interact.
      To avoid this, you should use the `copy` method of the `HMM`
      before making any changes to it.

    Once you've made some experiments you believe are noteworthy you should add a
    "suggested settings/tunings" section in comments at the bottom of
    `my_model/my_settings_1.py`, listing some of the relevant DA method
    configurations that you tested, along with the RMSE (or other stats) that
    you obtained for those methods.  You will find plenty of examples already in
    DAPPER, used for cross-referenced with literature to verify the workings of DAPPER
    (and the reproducibility of publications).
