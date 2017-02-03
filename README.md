
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
as well as its tools to manage experimental settings, averages, and random numbers,
all of which assist in ensuring that the results are neither frivolous nor spurious.
It is open source, written in Python, and focuses on code readability;
this promotes the reproduction and dissemination of the underlying science,
and makes it easy to adapt and extend to further needs.
In summary, it is well suited for conducting fundamental DA research
as well as teaching purposes.

To some extent, DAPPER sacrifices efficiency to gain the advantages listed above.
I.e. it is not designed for the assimilation of real data in operational models.

The current documentation does not provide a tutorial;
new users should begin by looking at the code in `benchmark.py`
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

Model name  | Linear? | Phys.dim. | State len.  | # Lyap>=0 | Thanks to
----------- | ------- | --------- | ----------- | --------- | ----------
Lin. Advect.| Yes     | 1D        | 1000        |  51       | Evensen
Lorenz63    | No      | 0D        | 3           |  2+       | Lorenz/Sakov
Lorenz84    | No      | 0D        | 3           |  2+       | Lorenz/Raanes
Lorenz95    | No      | 1D        | 40          |  13+      | Lorenz/Sakov
LorenzXY    | No      | 2x 1D     | 256 + 8     |  ≈13      | Lorenz/Raanes
MAOOAM      | No      | 2x 1D     | 36          |  ?        | Tondeur/Vannitsen
Barotropic  | No      | 2D        | 256^2 ≈ 60k |  ?        | J.Penn/Raanes


Additional features
------------------------------------------------
Many
* visualizations 
* diagnostics


And:
* Highly modular.
* Parallelizable
    * Forecast parallelization is possible since
        the (user-implemented) model has access to the full ensemble.
    * A light-weight alternative (see e.g. Lorenz95):
        native numpy vectorization (again by having access to full ensemble).
    * (Independent) experiments can also run in parallel.
        Auto-config provided by `utils.py:parallelize()`.


Also has:
* Progressbar
* Confidence interval on times series (e.g. rmse) with
	* automatic correction for autocorrelation 
	* significant digits printing
* X-platform random number generator
* Chronology/Ticker with consistency checks
* CovMat class (input flexibility/overloading, lazy eval)
* Live plotting with on/off toggle
* Intelligent defaults (e.g. plot duration estimated from autocorrelation,
    axis limits estimated from percentiles)


Alternative projects
------------------------------------------------
Sorted by approximate project size.
DAPPER may be situated somewhere in the middle.

* DART         (NCAR)
* SANGOMA      (Liege/CNRS/Nersc/Reading/Delft)
* PDAF         (Nerger)
* ERT*         (Statoil)
* OpenDA       (TU Delft)
* MIKE         (DHI)
* PyOSSE       (Edinburgh)
* FilterPy     (R. Labbe)
* PyIT         (CIPR)
* Datum*       (Raanes)
* EnKF-Matlab* (Sakov)
* IEnKS code*  (Bocquet)
* pyda         (Hickman)

*Has been inspirational in the development of DAPPER. 


How to
------------------------------------------------
DAPPER is like a *set of templates* (not a framework);
do not hesitate make your own scripts and functions
(instead of squeezing everything into standardized configuration files).

#### Add a new method
Just add it to `da_algos.py`, using the others in there as templates.
(TODO: split `da_algos.py` into multiple files.)


#### Add a new model
* Make a new dir: `DAPPER/mods/`**your_mod**
    * Remember to include the empty file `__init__.py`
    * See other examples, e.g. `DAPPER/mods/Lorenz63/sak12.py`
* Make sure that your model (and obs operator) support
    * 2D-array (i.e. ensemble) and 1D-array (single realization) input
        (can typically be handled by `@atmost_2d` wrapper).
    * should not modify in-place.
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


What it can't do
------------------------------------------------
* Run different DA methods concurrently (i.e. step-by-step)
     allowing for live/online  (graphic or text) comparison
* Time-dependent noises and length changes in state/obs
     (but it does support autonomous f and h)
* Non-uniform time sequences

TODO
------------------------------------------------
* CovMat
     * Unify sparse and dense treatment
     * Read-only properties
* Models
    * Improve doc
    * Barotropic
    * conversational 1D model (aside from L95)
    * KdVB (Zupanski 2006)


"Outreach"
---------------
* http://stackoverflow.com/a/38191145/38281
* http://stackoverflow.com/a/37861878/38281
