
     ___   _   ___ ___ ___ ___ 
    |   \ /_\ | _ \ _ \ __| _ \
    | |) / _ \|  _/  _/ _||   /
    |___/_/ \_\_| |_| |___|_|_\


![EnKF - Lorenz'63](./figs/l63_ens_animated.gif)


 - Author: Patrick N. Raanes
 - Licence: See licence.txt

Installation:
============================================
Requires python3.5, with scipy.

Then, download, and run:

    > python benchmarks.py

Features:
============================================
Many
 - methods
 - models
 - diagnostics




- Highly modular
- Highly modular

Efficient.
E.g. Lorenz-96 uses native vectorization (i.e. fast numpy),  but no parallelization.

Very explicit and readable.

Consistency checks (e.g. time).


What it can't do:
============================================
 - Store full ensembles (could write to file)
 - Run different DA methods concurrently (i.e. step-by-step)
     allowing for online (visual or console) comparison
 - Time-dependent noises and length changes in state/obs
     (but it does support autonomous f and h)
 - Non-uniform time sequences


Sugar:
============================================
 - Progressbar
 - Confidence interval on times series with automatic correction for its autocorrelation (e.g. rmse)
 - X-platform random number generator
 - Chronology/Ticker
 - CovMat class (input flexibility, / overloading, lazy eval)


Alternatives:
============================================
### Big
 - DART        (NCAR)
 - SANGOMA     (Liege/CNRS/Nersc/Reading/Delft)
 - PDAF        (Nerger)
 - ERT         (Statoil)
 - OpenDA      (TU Delft)
 - PyOSSE      (Edinburgh)
 - ?           (DHI)

#### Medium
 - FilterPy    (R. Labbe)
 - PyIT        (CIPR)
 - Datum       (Raanes)
    
#### Small
 - EnKF-Matlab (Sakov)
 - IEnKS code  (Bocquet)
 - pyda        (Hickman)


Implementation choices:
============================================
 - Uses python version >= 3.5
 - On-line vs off-line stats and diagnostics
 - Matrix class for syntax and broadcasting


##### Models f, h
 - to be defined in model file
 - should not modify in-place.
 - should take ensembles as input.
   (hence forecast parallelization is in users's hands)


Changelog / Fixes:
============================================
 - Circular import (when using common * in all files)?
 - Forgot sqrt(dt) for mod noise
 - array vs matrix broadcasting
 - Reproduce matlab: +D or -D (obs pert.)
 - Reproduce matlab: myrandn in RVs and EnsUpdateGlob
 - Reproduce matlab: reset LCG seed after EnsUpdateGlob
 - Migrating from global variables to parameter lists (classes) 
 - Matrix --> Array: Too troublesome to always assert that
   we're dealing with matrices coz numpy tends to return arrays
 - Row vs column-major ordering:
   By conforming to python's row-major ordering
   with the ensemble (i.e. one row <--> one member), we get
   - correct ordering random samples
     (i.e. adding a new member doesn't change previous draws) 
   - correct broadcasting with ensemble manipulations
   - avoid reshaping yy[:,k]
 - OLD: Mathematical (and standard) litterature var. names (i.e. h, R, P, E, A)
 - OLD: Many global namespace imports: from <package> import func1, func2



