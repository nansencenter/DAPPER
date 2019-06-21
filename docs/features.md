Additional features
================================================
* Progressbar
* Tools to manage and display experimental settings and stats
* Visualizations, including
    * liveplotting (during assimilation)
    * intelligent defaults (axis limits, ...)
* Diagnostics and statistics with
    * Confidence interval on times series (e.g. rmse) averages with
        * automatic correction for autocorrelation 
        * significant digits printing
* Parallelisation:
    * (Independent) experiments can run in parallel; see `example_3.py`
    * Forecast parallelisation is possible since
        the (user-implemented) model has access to the full ensemble;
        see example in `mods/QG/core.py`.
    * Analysis parallelisation over local domains;
        see example in `da_methods.py:LETKF()`
    * Also, numpy does a lot of parallelization when it can.
        However, as it often has significant overhead,
        this has been turned off (see `tools/utils.py`)
        in favour of the above forms of parallelization.
* Gentle failure system to allow execution to continue if experiment fails.
* Classes that simplify treating:
    * Time sequences Chronology/Ticker with consistency checks
    * random variables (`RandVar`): Gaussian, Student-t, Laplace, Uniform, ...,
      as well as support for custom sampling functions.
    * covariance matrices (`CovMat`): provides input flexibility/overloading, lazy eval) that facilitates the use of non-diagnoal covariance matrices (whether sparse or full).


.. Also has:
.. * X-platform random number generator (for debugging accross platforms)



