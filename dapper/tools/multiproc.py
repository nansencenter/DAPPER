"""Paralellisation via multiprocessing. Wraps pool.map for convenience."""

import functools

# Multiprocessing requries pickling. The package 'dill' is able to
# pickle much more than basic pickle (e.g. nested functions),
# and is being used by 'multiprocessing_on_dill',
# and the better-maintained pathos
import multiprocessing_on_dill as mpd
# Enforcing individual core usage.
# ---------
# Issue: numpy uses multiple cores (github.com/numpy/numpy/issues/11826).
#   This may yield some performance gain, but typically not much
#   compared to manual parallelization over independent experiments,
#   ensemble forecasts simulations, or local analyses.
# Therefore: force numpy to only use a single core.
#
# Ref https://stackoverflow.com/a/53224849
# for envar in [
#     "OMP_NUM_THREADS",        # openmp
#     "OPENBLAS_NUM_THREADS",   # openblas
#     "MKL_NUM_THREADS",        # mkl
#     "VECLIB_MAXIMUM_THREADS", # accelerate
#     "NUMEXPR_NUM_THREADS"]:   # numexpr
#     os.environ[envar] = "1"
# The above may be the safest way to limit thread use on all systems,
# but requires importing before np. => Instead, use threadpoolctl!
#
# >>> import numpy as np
# >>> from threadpoolctl import threadpool_limits
# >>> N  = 4*10**3
# >>> a  = np.random.randn(N, N)
# >>> # Now start monitoring CPU usage (with e.g. htop).
# >>> with threadpool_limits(limits=1, user_api='blas'):
# >>>   a2 = a @ a
import threadpoolctl

# Deciding on core numbers
# ---------
# Unnecessary. Just use NPROC=None.
# from psutil import cpu_percent, cpu_count

threadpoolctl.threadpool_limits(1)


def map(func, xx, **kwargs):  # noqa
    """A parallelized version of map.

    Similar to `result = [func(x, **kwargs) for x in xx]`, but also deals with:

    - passing kwargs
    - join(), close()
    - KeyboardInterrupt (not any more)

    Note: in contrast to reading operations, writing "in-place"
    does not work with multiprocessing. This changes with
    "shared" arrays, but this has not been tried out here.
    By contrast, multithreading shares the memory,
    but was significantly slower in the tested (pertinent) cases.

    NB: multiprocessing does not mix with matplotlib,
        so ensure `func` does not reference `self.stats.LP_instance`,
        where `self` is a `@da_method` object.
        In fact, `func` should not reference `self` at all,
        because its serialization is rather slow.

    See example use in `dapper.mods.QG`
    """
    NMAX = mpd.cpu_count() - 1  # Be nice
    NPROC = kwargs.pop("NPROC", NMAX)
    pool = mpd.Pool(NPROC)

    try:
        f = functools.partial(func, **kwargs)  # Fix kwargs

        # map vs imap: https://stackoverflow.com/a/26521507
        result = pool.map(f, xx)

    except Exception:
        pool.terminate()
        pool.close()
        pool.join()
        raise

    return result
