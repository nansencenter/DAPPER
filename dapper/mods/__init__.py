"""Models included with DAPPER.

The models are all simple;
this facililates the reliability, reproducibility, and transparency
of DA experiments.

See the README section on
[test cases (models)](https://github.com/nansencenter/DAPPER#Test-cases-models)
for an overview of the models included with DAPPER.

## Defining your own model

Follow the example of one of the models within the `dapper/mods` folder.
Essentially, you just need to define all of the attributes of a
`dapper.admin.HiddenMarkovModel`.
To make sure this is working, we suggest the following structure:

- Make a dir: `my_model`
- Make a file: `my_model/__init__.py` where you define the core
  workings of the model.
  Typically, this culminates in a `step(x, t, dt)` function.
    - The model step operator (and the obs operator) must support
      2D-array (i.e. ensemble) and 1D-array (single realization) input.
      See `dapper.mods.Lorenz63` and `dapper.mods.Lorenz96`
      for typical implementations,
      and `dapper.mods/QG` for how to parallelize the ensemble simulations.
    - Optional: To use the (extended) Kalman filter, or 4D-Var,
      you will need to define the model linearization.
      Note: this only needs to support 1D input (single realization).
- Make a file: `my_model/demo.py` to visually showcase
  a simulation of the model.
- Make a file: `my_model/my_settings_1.py` that define a complete
  Hidden Markov Model ready for a synthetic experiment
  (also called "twin experiment" or OSSE).


<!--
* To begin with, test whether the model works
    * on 1 realization
    * on several realizations (simultaneously)
* Thereafter, try assimilating using
    * a big ensemble
    * a safe (e.g. 1.2) inflation value
    * small initial perturbations
      (big/sharp noises might cause model blow up)
    * small(er) integrational time step
      (assimilation might create instabilities)
    * very large observation noise (free run)
    * or very small observation noise (perfectly observed system)
-->
"""
