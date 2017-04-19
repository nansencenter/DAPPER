
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
through its variety of typical test cases and statistics.
DAPPER (1) reproduces numerical results (benchmarks) reported in the literature,
and (2) facilitates comparativ studies through its collection of baseline methods
and its tools to manage experimental settings, averages, and random numbers.
The above assist in ensuring that the results are (1) reliable and (2) relevant.

It is open source, written in Python, and focuses on code readability;
this promotes the reproduction and dissemination of the underlying science,
and makes it easy to adapt and extend to further needs.
In summary, it is well suited for conducting fundamental DA research
as well as for teaching purposes.

In a trade-off with the above advantages, DAPPER makes some sacrifices of efficiency and flexibility (generality).
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

Method name                                            | Literature RMSE results reproduced
------------------------------------------------------ | ---------------------------------------
EnKF <sup>1</sup>                                      | sakov'2008 ("deterministic")
EnKF-N                                                 | bocquet'2012 ("combining"), bocquet'2015 ("expanding")
EnKS, EnRTS                                            | raanes'2016 ("EnRTS and EnKS")
iEnKF                                                  | sakov'2012 ("an iterative")
LETKF, local & serial EAKF                             | bocquet'2011 ("EnKF-N")
Sqrt. model noise methods                              | raanes'2015 ("sqrt model noise")
Particle filter (bootstrap & implicit) <sup>2</sup>    | bocquet'2010 ("beyond Gaussian")
Extended KF                                            | raanes'2016 thesis
3D-Var                                                 | "
Climatology                                            | "

<sup>1</sup>: Stochastic, DEnKF (i.e. half-update), ETKF (i.e. sym. sqrt.).  
Tuned with inflation and "random, orthogonal rotations".  
<sup>2</sup>: Resampling: multinomial (including systematic and residual).  
The particle filter is tuned with "effective-N monitoring", "regularization", "adjusted resampling weights", "annealed prior".  


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
* CovMat class (input flexibility/overloading, lazy eval) that facilitates
    the use of non-diagnoal covariance matrices (whether sparse or full)
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

<!--
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
* OAK          (Liège)
* Siroco       (OMP)
* Datum*       (Raanes)
* EnKF-Matlab* (Sakov)
* IEnKS code*  (Bocquet)
* pyda         (Hickman)

*: Has been inspirational in the development of DAPPER. 
-->


Name               | Developers           | Purpose (vs. DAPPER)
------------------ | -------------------- | -----------------------------
[DART][1]          | NCAR                 | Operational and real-world DA
[ERT][2]*          | Statoil              | Operational (petroleum) history matching
[OpenDA][3]        | TU Delft             | Operational and real-world DA
[EMPIRE][4]        | Reading (Met)        | Operational and real-world DA
[SANGOMA][5]       | Conglomerate**       | Unified code repository researchers
[Verdandi][6]      | INRIA                | Real-world biophysical DA
[PDAF][7]          | Nerger               | Real-world and example DA
[PyOSSE][8]        | Edinburgh, Reading   | Real-world earth-observation DA
[MIKE][9]          | DHI                  | Real-world oceanographic DA. Commercial?
[OAK][10]          | Liège                | Real-world oceaonagraphic DA
[Siroco][11]       | OMP                  | Real-world oceaonagraphic DA
[FilterPy][12]     | R. Labbe             | Engineering, general intro to Kalman filter
[DASoftware][13]   | Yue Li, Stanford     | Matlab, large-scale
[Pomp][18]         | U of Michigan        | R, general state-estimation
[PyIT][14]         | CIPR                 | Real-world petroleum DA (?)
Datum*             | Raanes               | Matlab, personal publications
[EnKF-Matlab*][15] | Sakov                | Matlab, personal publications and intro
[EnKF-C][17]       | Sakov                | C, light-weight EnKF, off-line
IEnKS code*        | Bocquet              | Python, personal publications
[pyda][16]         | Hickman              | Python, personal publications


<!--
Real-world: supports very general models (e.g. time dependent state length, mapping to-from grids, etc.)
Operational: optimized for speed.
-->

*: Has been inspirational in the development of DAPPER. 

**: Liege/CNRS/NERSC/Reading/Delft

[1]: http://www.image.ucar.edu/DAReS/DART/
[2]: http://ert.nr.no/ert/index.php/Main_Page
[3]: http://www.openda.org/
[4]: http://www.met.reading.ac.uk/~darc/empire/index.php
[5]: http://www.data-assimilation.net/
[6]: http://verdandi.sourceforge.net/
[7]: http://pdaf.awi.de/trac/wiki
[8]: http://www.geos.ed.ac.uk/~lfeng/
[9]: http://www.dhigroup.com/
[10]: http://modb.oce.ulg.ac.be/mediawiki/index.php/Ocean_Assimilation_Kit
[11]: https://www5.obs-mip.fr/sirocco/assimilation-tools/sequoia-data-assimilation-platform/
[12]: https://github.com/rlabbe/filterpy
[13]: https://github.com/judithyueli/DASoftware
[14]: http://uni.no/en/uni-cipr/
[15]: http://enkf.nersc.no/
[16]: http://hickmank.github.io/pyda/
[17]: https://github.com/sakov/enkf-c
[18]: https://github.com/kingaa/pomp


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
* Make Operator callable (`--> .model`) `and __repr__`
* Insert `fix_mu` and `fix_var` in center
* Make lightweight OOP assimilation methods:
    - replace config by function arguments
    - return function 'assimilate'
    - wrapped to catch exceptions and return stats
* Split `da_algos.py` into multiple files
* Make full 3D-Var (not opt. int.)
* Make tutorial



<!--
"Outreach"
---------------
* http://stackoverflow.com/a/38191145/38281
* http://stackoverflow.com/a/37861878/38281
-->

