
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

Installation:
------------------------------------------------
Requires python3.5 with scipy.

Then, download DAPPER, and run:

    > python benchmarks.py

Features:
------------------------------------------------
Reproduces benchmark results from
* sakov'2008 (ETKF,DEnKF,EnKF) with LA model and Lorenz'96
* sakov'2012 (approximately iETKF)
* bocquet'2012 (EnKF-N)
* raanes'2014 (Sqrt model noise methods)	
* raanes'2016 (Thesis: Particle filter, ExtKF, 3D-Var)
* TODO: raanes'2015 (EnKS vs EnRTS)	
* TODO: bocquet'2014 (EnKS-N)
* TODO: bocquet'2015 (EnKF-N)

Many
* methods
* models
* diagnostics


Highly modular.

Efficient
E.g. Lorenz-96 uses native vectorization (i.e. fast numpy),  but no parallelization.

Very explicit and readable.

Consistency checks (e.g. time).

For -N stuff, compared to Boc's code, DAPPER
* uses more matrix decompositions (efficiency),
* allows for non-diag R.


What it can't do:
------------------------------------------------
* Store full ensembles (could write to file)
* Run different DA methods concurrently (i.e. step-by-step)
     allowing for online (visual or console) comparison
* Time-dependent noises and length changes in state/obs
     (but it does support autonomous f and h)
* Non-uniform time sequences


Sugar:
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


Alternatives:
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


Implementation choices:
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


How to make a new experiment
------------------------------------------------
* For understanding: view "Perfect Model Experiment Overview" section of http://www.image.ucar.edu/DAReS/DART/DART_Starting.php

How to insert new model
------------------------------------------------
* Make a new dir: DAPPER/mods/**your_mod**
* See other examples, e.g. DAPPER/mods/L3/sak12.py
* Make sure that your model (and obs operator) support
  **ensemble input**
* To begin with, try **small** initial perturbations.
  Big and sharp (white) might cause your model to blow up!

##### Models f, h
* should not modify in-place.
* should take ensembles as input.
   (hence forecast parallelization is in users's hands)

TODO
------------------------------------------------
* LA model
* PartFilt
* ExtKF
* Climatology
* Localization
* add_noise()
* 1D model from workshop that preserves some quantity
* 2D model
* average obs and truth rank hist
* iEnKS-N
* Models come with their own viz specification
* Toggle LivePlot **on Windows**
* Should observations return copy? e.g. x[:,obsInds].copy()
* 
* Add before/after analysis plots
* 
* Truncate SVD at 95 or 99% (evensen)
* unify matrix vs array (e.g. randn)
* vs 1d array (e.g. xx[:,0] in L3.dxdt)
* avoid y  = yy[:,kObs].reshape((p,1))
* Take advantage of pass-by-ref
* Decide on conflicts np vs math vs sp
* prevent CovMat from being updated
* Doc models



Changelog / Fixes:
------------------------------------------------
* Circular import (when using common * in all files)?
* Forgot sqrt(dt) for mod noise
* array vs matrix broadcasting
* Reproduce matlab: +D or -D (obs pert.)
* Reproduce matlab: myrandn in RVs and EnsUpdateGlob
* Reproduce matlab: reset LCG seed after EnsUpdateGlob
* Migrating from global variables to parameter lists (classes) 
* Matrix --> Array: Too troublesome to always assert that
   we're dealing with matrices coz numpy tends to return arrays
* Row vs column-major ordering:
  * By conforming to python's row-major ordering
  with the ensemble (i.e. one row <--> one member), we get
  correct ordering random samples
  (i.e. adding a new member doesn't change previous draws) 
  * correct broadcasting with ensemble manipulations
     * avoid reshaping yy[:,k]
* OLD: Mathematical (and standard) litterature var. names (i.e. h, R, P, E, A)
* OLD: Many global namespace imports: from <package> import func1, func2


Outreach:
---------------
* http://stackoverflow.com/a/38191145/38281
* http://stackoverflow.com/a/37861878/38281
