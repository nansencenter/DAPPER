Implementation specifics
========================

Choices
-------------

* Uses python3.5+
* NEW: Use `N-by-Nx` ndarrays. Pros:
    * python default
        * speed of (row-by-row) access, especially for models
        * same as default ordering of random numbers
    * facilitates typical broadcasting
    * less transposing for for ens-space formulae
    * beneficial operator precedence without `()`. E.g. `dy @ Rinv @ Y.T @ Pw` (where `dy` is a vector)
    * fewer indices: `[n,:]` yields same as `[n]`
    * no checking if numpy return `ndarrays` even when input is `matrix`
    * Regression literature uses `N-by-Nx` ("data matrix")
* OLD: Use `Nx-by-N` matrix class. Pros:
    * EnKF literature uses `Nx-by-N`
    * Matrix multiplication through `*` -- since python3.5 can just use `@`

Conventions
---------------

* DA_Config, assimilate, stats
* fau_series
* E,w,A
* Nx-by-N
* Nx (not ndims coz thats like 2 for matrices), P, chrono
* n
* ii, jj
* xx,yy
* no obs at 0



Notations
-------------
* xp: an experiment configuration. Includes...
* xpList: list of xps
* xpCube: dict of xps, perceived as a high-dim. sparse matrix
* x: truth
* y: obs
* E: ensemble -- shape (N,M)
* xn: member (column) n of E
* M: ndims
* N: ensemble size (number of members)
* Nx: ndims x
* Ny: ndims y
* Repeat symbols: series
* xx: time series of truth -- shape (K+1, M)
* kk: time series of times
* yy: time series of obs
* EE: time series of ens
