
     ___   _   ___ ___ ___ ___ 
    |   \ /_\ | _ \ _ \ __| _ \
    | |) / _ \|  _/  _/ _||   /
    |___/_/ \_\_| |_| |___|_|_\


![EnKF - Lorenz'63](./figs/Lor63_ens_anim_2.gif)

* DAPPER is a platform for "twin experiments", where you
  * Specify your model (must support ensemble input)
  * Generate a synthetic truth
  * Generate synthetic obs
  * Benchmark different DA methods
* Developed at the Nansen centre. Contributors:
  * Patrick N. Raanes
  * Maxime Tondeur
* Licence: See licence.txt

Installation
------------------------------------------------
Requires python3.5 with scipy.

Then, download DAPPER, and run:

    > python benchmarks.py

Methods
------------
* EnKF (Perturbed-Obs,ETKF,DEnKF)
* EnKF-N
* EnKS
* iterative versions of the above
    (as in Bocquet/Sakov litterature)
* Extended KF
* Particle filter (bootstrap)
* 3D-Var
* Climatology


Models
------------

Model name  | Linear? | Phys.dim. | State len. | Model subspace dim.
----------- | ------- | --------- | ---------- | ---------------------
LA          | Yes     | 1D        |  1000      |  51
Lorenz63    | No      | 0D        |  3         |  2+
Lorenz95    | No      | 1D        |  40        |  13+
Lorenz95_2s | No      | 2x 1D     |  256 + 8   |  ?
MAOOAM      | No      | 2x 1D     |  36        |  ?


#### How to add a new model
* Make a new dir: DAPPER/mods/**your_mod**
* See other examples, e.g. DAPPER/mods/Lorenz63/sak12.py
* Make sure that your model (and obs operator) support
    * **ensemble input**
      (allowing forecast parallelization is in users's hands)
    * should not modify in-place.
    * the same applies for the observation operator/model
* To begin with, try **small** initial perturbations.
  Big and sharp (white) might cause your model to blow up!
* Nice read: "Perfect Model Experiment Overview" section of
    http://www.image.ucar.edu/DAReS/DART/DART_Starting.php


Features
------------------------------------------------
Many
* visualizations 
* diagnostics


Also:
* Highly modular.
* Balance between efficiency and readability.
* Consistency checks (e.g. time).

<!---
E.g. Lorenz-96 uses native vectorization (i.e. fast numpy),
  but no parallelization.
-->

<!---
For -N stuff, compared to Boc's code, DAPPER
* uses more matrix decompositions (efficiency),
* allows for non-diag R.
-->


Sugar
------------------------------------------------
* Progressbar
* Confidence interval on times series (e.g. rmse) with
	* automatic correction for autocorrelation 
	* significant digits printing
* X-platform random number generator
* Chronology/Ticker
* CovMat class (input flexibility, / overloading, lazy eval)
* Live plotting with on/off toggle
* Intelligent defaults (e.g. plot duration estimated from acf,
    axis limits esitmated from percentiles)


What it can't do
------------------------------------------------
* Store full ensembles (could write to file)
* Run different DA methods concurrently (i.e. step-by-step)
     allowing for online (visual or console) comparison
* Time-dependent noises and length changes in state/obs
     (but it does support autonomous f and h)
* Non-uniform time sequences


Implementation choices
------------------------------------------------
* Uses python version >= 3.5
* On-line vs off-line stats and diagnostics
* NEW: Use N-by-m ndarrays. Pros:
    * Python default
        * speed of (row-by-row) access, especially for models
        * ordering of random numbers
    * numpy sometimes returns ndarrays even when input is matrix
    * works well with ens space formulea,
        * e.g. 
        * yields beneficial operator precedence without (). E.g. dy@Ri@Y.T@Pw
    * Bocquet's choice
    * Broadcasting
    * Avoids reshape's and asmatrix
* OLD: Use m-by-N matrix class. Pros:
    * Litterature uses m-by-N
    * Matrix class allowss desired broadcasting
    * Deprecated: syntax (* vs @)


Alternatives
------------------------------------------------
##### Big
* DART        (NCAR)
* SANGOMA     (Liege/CNRS/Nersc/Reading/Delft)
* PDAF        (Nerger)
* ERT         (Statoil)
* OpenDA      (TU Delft)
* PyOSSE      (Edinburgh)
* ?           (DHI)

##### Medium
* FilterPy    (R. Labbe)
* PyIT        (CIPR)
* Datum       (Raanes)
    
##### Small
* EnKF-Matlab (Sakov)
* IEnKS code  (Bocquet)
* pyda        (Hickman)


TODO
------------------------------------------------
* iEnKS-N
* Localization
* add_noise()
* before/after analysis viz
* 1D model from workshop that preserves some quantity
* 2D model
* Doc models

* Should (direct) observations return copy? e.g. x[:,obsInds].copy()
* Take advantage of pass-by-ref
* Decide on conflicts np vs math vs sp

* Truncate SVD at 95 or 99% (evensen)
* unify matrix vs array (e.g. randn)
* vs 1d array (e.g. xx[:,0] in L3.dxdt)
* prevent CovMat from being updated


"Outreach"
---------------
* http://stackoverflow.com/a/38191145/38281
* http://stackoverflow.com/a/37861878/38281
