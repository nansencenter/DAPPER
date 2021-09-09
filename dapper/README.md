## Installation

See [README/Installation](https://github.com/nansencenter/DAPPER#Installation)

## Usage

See [README/Getting-started](https://github.com/nansencenter/DAPPER#Getting-started)

Then, adapt one of the
[example scripts](https://github.com/nansencenter/DAPPER/tree/master/examples)
to your needs. Make sure you have a working adaptation `examples/basic_1.py`
before adapting `basic_2` and `basic_3`.

Since the generality of DAPPER is
[limited](https://github.com/nansencenter/DAPPER#similar-projects)
it is quite likely you will also need to make changes to the DAPPER code itself.

If, in particular, you wish to illustrate and run benchmarks with
your own **model** or **method**, then

- If it is a complex one, you may be better off using DAPPER
  merely as *inspiration* (but you can still
  [cite it](https://github.com/nansencenter/DAPPER#getting-started))
  rather than trying to squeeze everything into its templates.
- If it is relatively simple, however, you may well want to use DAPPER.
  In that case, read this:
    - `mods`
    - `da_methods`

### Developer guide

If you are making a pull request, please read the `dev_guide`.

## Bibliography/references

See `bib`.

## Features

The main features are listed in
[README/highlights](https://github.com/nansencenter/DAPPER#Highlights).
Additionally, there is:

- Parallelisation:
    - (Independent) experiments can run in parallel; see `examples/basic_3.py`
    - Forecast parallelisation is possible since
        the (user-implemented) model has access to the full ensemble;
        see example in `mods.QG`.
    - Analysis parallelisation over local domains;
        see example in `da_methods.ensemble.LETKF`
- Classes that simplify treating:
    - Experiment administration and launch via `xp_launch.xpList`
      and data processing and presentation via `xp_process.xpSpace`.
    - Time sequences use via `tools.chronos.Chronology`
      and`tools.chronos.Ticker`.
    - Random variables via `tools.randvars.RV`: Gaussian, Student-t, Laplace, Uniform,
      ..., as well as support for custom sampling functions.
    - Covariance matrices via `tools.matrices.CovMat`:
      provides input flexibility/overloading,
      lazy eval that facilitates the use of non-diagonal
      covariance matrices (whether sparse or full).
- Diagnostics and statistics with
    - Confidence interval on times series (e.g. rmse) averages with
        - automatic correction for autocorrelation
        - significant digits printing
    - Automatic averaging of several types for sub-domains
      (e.g. "ocean", "land", etc.)

## API reference

The rendered docstrings can be browsed
through the following links, which are also available in the left sidebar.
