
<!--
!      ___   _   ___ ___ ___ ___ 
!     |   \ /_\ | _ \ _ \ __| _ \
!     | |) / _ \|  _/  _/ _||   /
!     |___/_/ \_\_| |_| |___|_|_\
! 
! 
-->

DAPPER is a set of templates for benchmarking the performance of
[data assimilation (DA)](https://sites.google.com/site/patricknraanespro/DA_tut.pdf)
methods.
The tests provide experimental support and guidance for new developments in DA.
The screenshot below illustrates the default diagnostics.

<!--
![EnKF - Lorenz'63](./data/figs/anims/Lor63_ens_anim_2.gif)
-->

![EnKF - Lorenz'63](./data/figs/anims/DAPPER_illust_v2.jpg)


The typical set-up is a "twin experiment", where you
* specify a
	* dynamic model 
	* observational model 
* use these to generate a synthetic
	* "truth"
	* and observations thereof
* assess how different DA methods
	perform in estimating the truth

DAPPER makes the numerical investigation of DA methods accessible
through its variety of typical test-cases and statistics.
It reproduces numerical results reported in the literature,
which safeguards the quality of its benchmarks.
Comparative studies are facilitated by its collection of baseline methods
and its tools to manage experimental settings, averages, and random numbers,
all of which assist in ensuring that the results are neither frivolous nor spurious.
It is open source, written in Python, and focuses on code readability;
this promotes the reproduction and dissemination of the underlying science,
and makes it easy to adapt and extend to further needs.
In summary, it is well suited for conducting fundamental DA research
as well as for teaching purposes.

In a trade-off with the above advantages, DAPPER makes some sacrifices of efficiency.
I.e. it is not designed for the assimilation of real data in operational models.

The current documentation does not provide a tutorial;
new users should begin by looking at the code in `bench_example.py`
and work their way backward.

	
Installation
------------------------------------------------
Prerequisite: python3.5+ with scipy (e.g. from [anaconda](https://www.continuum.io/downloads))

Download, extract, and `cd` to DAPPER. Then run:

    > python -i benchmark.py

Methods
------------

Method name                        | Literature RMSE results reproduced
-----------------------------------| ---------------------------------------
EnKF (Stoch., DEnKF, ETKF)         | sakov'2008 ("deterministic")
EnKF-N                             | bocquet'2012 ("combining"), bocquet'2015 ("expanding")
EnKS, EnRTS                        | raanes'2016 ("EnRTS and EnKS")
Iterative versions of the above    | sakov'2012 ("an iterative"), TODO: bocquet'2014
LETKF, local & serial EAKF         |
Sqrt. model noise methods          | raanes'2015 ("sqrt model noise")
Extended KF                        | raanes'2016 thesis
Particle filter (bootstrap)        | "
3D-Var                             | "
Climatology                        | "


Models
------------

Model       | Linear? | Phys.dim. | State len | # Lyap≥0 | Thanks to
----------- | ------- | --------- | --------- | -------- | ----------
Lin. Advect.| Yes     | 1D        | 1000 *    | 51       | Evensen/Raanes
Lorenz63    | No      | 0D        | 3         | 2        | Lorenz/Sakov
Lorenz84    | No      | 0D        | 3         | 2        | Lorenz/Raanes
Lorenz95    | No      | 1D        | 40 *      | 13       | Lorenz/Raanes
LorenzXY    | No      | 2x 1D     | 256 + 8 * | ≈13      | Lorenz/Raanes
MAOOAM      | No      | 2x 1D     | 36        | ?        | Tondeur/Vannitsem
Quasi-Geost | No      | 2D        | 129²≈17k  | ≈135     | Sakov
Barotropic  | No      | 2D        | 256²≈60k  | ?        | J.Penn/Raanes

*: straightforward to vary.


Additional features
------------------------------------------------
Many
* Visualizations 
* Diagnostics


Also has:
* Live plotting with on/off toggle
* Confidence interval on times series (e.g. rmse) with
	* automatic correction for autocorrelation 
	* significant digits printing
* CovMat class (input flexibility/overloading, lazy eval)
* Intelligent defaults (e.g. plot duration estimated from autocorrelation,
    axis limits estimated from percentiles)
* Chronology/Ticker with consistency checks
* Progressbar
* X-platform random number generator
* Parallelisation options
    * Forecast parallelisation is possible since
        the (user-implemented) model has access to the full ensemble
        (see `mods/QG/core.py`)
    * A light-weight alternative (see e.g. `mods/Lorenz95/core.py`):
        native numpy vectorization (again by having access to full ensemble).
    * (Independent) experiments can also run in parallel.
        Auto-config provided by `utils.py:parallelize()`.


How to
------------------------------------------------
DAPPER is like a *set of templates* (not a framework);
do not hesitate make your own scripts and functions
(instead of squeezing everything into standardized configuration files).

#### Add a new method
Just add it to `da_algos.py`, using the others in there as templates.


#### Add a new model
* Make a new dir: `DAPPER/mods/`**your_mod**
* Remember to include the empty file `__init__.py`
* See other examples, e.g. `DAPPER/mods/Lorenz63/sak12.py`
* Make sure that your model (and obs operator) supports
  2D-array (i.e. ensemble) and 1D-array (single realization) input.
  See `Lorenz63` and `Lorenz95` for typical implementation.



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



What it can't do
------------------------------------------------
* Do highly efficient DA on very big models (see discussion in introdution).
* Run different DA methods concurrently (i.e. step-by-step)
     allowing for live/online  (graphic or text) comparison.
* Time-dependent error coviariances and changes in lengths of state/obs
     (but f and h may otherwise depend on time).
* Non-uniform time sequences.


Alternative projects
------------------------------------------------
Sorted by approximate project size.
DAPPER may be situated somewhere in the middle.

* DART         (NCAR)
* SANGOMA      (Liege/CNRS/Nersc/Reading/Delft)
* Verdandi     (INRIA)
* PDAF         (Nerger)
* ERT*         (Statoil)
* OpenDA       (TU Delft)
* MIKE         (DHI)
* PyOSSE       (Edinburgh)
* FilterPy     (R. Labbe)
* DASoftware   (Yue Li, Stanford)
* PyIT         (CIPR)
* Datum*       (Raanes)
* EnKF-Matlab* (Sakov)
* IEnKS code*  (Bocquet)
* pyda         (Hickman)

*: Has been inspirational in the development of DAPPER. 


Implementation choices
------------------------------------------------
* Uses python3.5+
* NEW: Use `N-by-m` ndarrays. Pros:
    * Python default
        * speed of (row-by-row) access, especially for models
        * same as default ordering of random numbers
    * numpy *might* return `ndarrays` even when input is matrix
    * less transposing for for ens-space formulae
    * beneficial operator precedence without `()`. E.g. `dy @ Rinv @ Y.T @ Pw` (where `dy` is a vector)
    * Avoids reshape's and recasting (`asmatrix`)
    * Fewer indices: `[n,:]` yields same as `[n]`
* OLD: Use `m-by-N` matrix class. Pros:
    * Literature uses `m-by-N`
    * Facilitates desired broadcasting
    * Matrix multiplication through `*` -- since python3.5 can just use `@`


TODO
------------------------------------------------
* CovMat
     * Unify sparse and dense treatment
     * Read-only properties
* Complete QG, LorenzXY
* Split `da_algos.py` into multiple files
* Make tutorial


<!--
"Outreach"
---------------
* http://stackoverflow.com/a/38191145/38281
* http://stackoverflow.com/a/37861878/38281
-->

