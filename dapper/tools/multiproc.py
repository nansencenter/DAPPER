"""Paralellisation via multiprocessing. Limit num. of CPUs used by `numpy` to 1."""

# To facilitate debugging, import mp here (where it's used).
# For example, `matplotlib` (its event loop) might clash (however mysteriously)
# with mp, so it's nice to be able to be sure that no mp is "in the mix".
import multiprocessing_on_dill as mpd
# Enforcing individual core usage.
# Issue: numpy uses multiple cores (github.com/numpy/numpy/issues/11826).
#     This may yield some performance gain, but typically not much
#     compared to "manual" parallelization over
#     experiments, ensemble members, or local analyses.
#     Therefore: force numpy to only use a single core.
import threadpoolctl

threadpoolctl.threadpool_limits(1)
# Alternative: ref https://stackoverflow.com/a/53224849
# for envar in [
#     "OMP_NUM_THREADS",        # openmp
#     "OPENBLAS_NUM_THREADS",   # openblas
#     "MKL_NUM_THREADS",        # mkl
#     "VECLIB_MAXIMUM_THREADS", # accelerate
#     "NUMEXPR_NUM_THREADS"]:   # numexpr
#     os.environ[envar] = "1"
# The above may be the safest way to limit thread use on all systems,
# but must be imported ahead of numpy, which is clumsy.
#
# Test case:
# >>> import numpy as np
# ... from threadpoolctl import threadpool_limits
# ... N  = 4*10**3
# ... a  = np.random.randn(N, N)
# ... # Now start monitoring CPU usage (with e.g. htop).
# ... with threadpool_limits(limits=1, user_api='blas'):
# ...     a2 = a @ a


def Pool(NPROC=None):
    """Initialize a multiprocessing `Pool`.

    - Uses `pathos/dill` for serialisation.
    - Provides unified interface for multiprocessing on/off (as a function of NPROC).

    There is some overhead associated with the pool creation,
    so you likely want to re-use a pool rather than repeatedly creating one.
    Consider using `functools.partial` to fix kwargs.

    .. note::
        In contrast to *reading*, in-place writing does not work with multiprocessing.
        This changes with "shared" arrays, but that has not been tested here.
        By contrast, multi*threading* shares the process memory,
        but was significantly slower in the tested (pertinent) cases.

    .. caution::
        `multiprocessing` does not mix with `matplotlib`, so ensure `func` does not
        reference `xp.stats.LP_instance`. In fact, `func` should not reference `xp`
        at all, because it takes time to serialize.

    See example use in `dapper.mods.QG` and `dapper.da_methods.LETKF`.
    """
    if NPROC == False:
        # Yield plain old map
        class NoPool:
            def __enter__(self): return builtins
            def __exit__(self, *args): pass
        import builtins
        return NoPool()

    else:
        # from psutil import cpu_percent, cpu_count
        if NPROC in [True, None]:
            NPROC = mpd.cpu_count() - 1  # be nice

        return mpd.Pool(NPROC)
