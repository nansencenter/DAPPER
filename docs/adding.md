Adding models/methods
======================
Remember: DAPPER is a *set of templates* (not a framework);
do not hesitate make your own scripts and functions
(instead of squeezing everything into standardized configuration files).


Adding a new method
----------------------------------
Follow the example of one of the methods in the `da_methods` folder.


Adding a new model
----------------------------------

Make a new dir: `DAPPER/mods/your_model`. Add the following files:
* `core.py` to define the core functionality and documentation of your dynamical model.
    Typically this culminates in a `step(x, t, dt)` function.
  * The model step operator (and the obs operator) must support
    2D-array (i.e. ensemble) and 1D-array (single realization) input.
    See the `core.py` file in `mods/Lorenz63` and `mods/Lorenz95` for typical
    implementations, and `mods/QG` for how to parallelize the ensemble simulations.
  * Optional: To use the (extended) Kalman filter, or 4D-Var,
    you will need to define the model linearization.
    Note: this only needs to support 1D input (single realization).
* `demo.py` to visually showcase a simulation of the model.
* Files that define a complete Hidden Markov Model ready for a twin experiment (OSSE).
    For example, this will plug in the `step`function you made previously as in `Dyn['model'] = step`.
    For further details, see examples such as `DAPPER/mods/Lorenz63/{sak12,boc12}.py`.


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




