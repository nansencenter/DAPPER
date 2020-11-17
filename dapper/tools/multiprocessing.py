"""Paralellisation via multiprocessing. Wraps pool.map for convenience."""

import functools
import textwrap
import dapper.tools.utils as utils

# Multiprocessing is kept as an option in DAPPER,
# since some of these libraries are experimental.
# TODO 2: make it mandatory?
try:
    # Multiprocessing requries pickling. The package 'dill' is able to
    # pickle much more than basic pickle (e.g. nested functions),
    # and is being used by 'multiprocessing_on_dill'.
    # Alternatively, the package pathos also enables multiprocessing with dill.
    import multiprocessing_on_dill as mpd

    # Deciding on core numbers
    # from psutil import cpu_percent, cpu_count

    # Enforcing individual core usage.
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
    threadpoolctl.threadpool_limits(1)

    no_MP = False
except ImportError:
    no_MP = True

    @utils.do_once
    def MP_warn():
        print(textwrap.dedent("""
        Warning: Multiprocessing (MP) was requsted during execution,
        but has not been properly installed.
        Try re-installing DAPPER with `pip install -e <path-to-DAPPER>[MP]`.
        """))


#########################################
# Multiprocessing
#########################################

if no_MP:
    def map(func, xx, **kwargs):
        MP_warn()
        return [func(x, **kwargs) for x in xx]

else:
    def map(func, xx, **kwargs):
        """A parallelized version of map.

        Similar to::

        >>> result = [func(x, **kwargs) for x in xx]

        Note: unlike reading, writing "in-place" does not work with multiprocessing
        (unless "shared" arrays are used, but this has not been tried out here).

        NB: multiprocessing does not mix with matplotlib,
            so ensure ``func`` does not reference ``self.stats.LP_instance``,
            where ``self`` is a ``@da_method`` object.
            In fact, ``func`` should not reference ``self`` at all,
            because its serialization is rather slow.

        See example use in mods/QG/core.py.

        Technicalities dealt with:
         - passing kwargs
         - join(), close()

        However, the main achievement of this helper function is to make
        "Ctrl+C", i.e. KeyboardInterruption,
        stop the execution of the program, and do so "gracefully",
        something which is quite tricky to achieve with multiprocessing.
        """

        # The Ctrl-C issue is mainly cosmetic, but an annoying issue. E.g.
        #  - Pressing ctrl-c should terminate execution.
        #  - It should only be necessary to press Ctrl-C once.
        #  - The traceback does not extend beyond the multiprocessing management
        #    (not into the worker codes), and should therefore be cropped before then.
        #
        # NB: Here be fuckin dragons!
        # This solution is mostly based on https://stackoverflow.com/a/35134329
        # I don't (fully) understand why the issues arise,
        # nor why my patchwork solution somewhat works.
        #
        # I urge great caution in modifying this code because
        # issue reproduction is difficult (coz behaviour depends on
        # where the execution is currently at when Ctrl-C is pressed)
        # => testing is difficult.
        #
        # Alternative to try:
        # - Use concurrent.futures, as the bug seems to have been patched there:
        #   https://bugs.python.org/issue9205. However, does this work with dill?
        # - Multithreading: has the advantage of sharing memory,
        #   but was significantly slower than using processes,
        #   testing on DAPPER-relevant.

        # Ignore Ctrl-C.
        # Alternative: Pool(initializer=[ignore sig]).
        # But the following way seems to work better.
        import signal
        orig = signal.signal(signal.SIGINT, signal.SIG_IGN)

        # Setup multiprocessing pool (pool workers should ignore Ctrl-C)
        NPROC = None  # None => multiprocessing.cpu_count()
        pool = mpd.Pool(NPROC)

        # Restore Ctrl-C action
        signal.signal(signal.SIGINT, orig)

        try:
            f = functools.partial(func, **kwargs)  # Fix kwargs

            # map vs imap: https://stackoverflow.com/a/26521507
            result = pool.map(f, xx)

            # Relating to Ctrl-C issue, map_async was preferred:
            # https://stackoverflow.com/a/1408476
            # However, this does not appear to be necessary anymore...
            # result = pool.map_async(f, xx)
            # timeout = 60 # Required for get() to not ignore signals.
            # result = result.get(timeout)

        except KeyboardInterrupt:
            try:
                pool.terminate()
                # Attempts to propagate "Ctrl-C" with reasonable traceback print:
                # ------------------------------------------------------------------
                # ALTERNATIVE 1: ------- shit coz: only includes multiprocessing trace.
                # traceback.print_tb(e.__traceback__,limit=1)
                # sys.exit(0)
                # ALTERNATIVE 2: ------- shit coz: includes multiprocessing trace.
                # raise e
                # ALTERNATIVE 3: ------- shit coz: includes multiprocessing trace.
                # raise KeyboardInterrupt
                # ALTERNATIVE 4:
                was_interrupted = True
            except KeyboardInterrupt:
                # Sometimes the KeyboardInterrupt caught above
                # just causes things to hang, and another "Ctrl-C" is required,
                # which is then caught by this 2nd try-catch.
                pool.terminate()
                was_interrupted = True
        else:
            # Resume normal execution
            was_interrupted = False
            pool.close()  # => Processes will terminate once their jobs are done.

        try:
            # Helps with debugging,
            # according to https://stackoverflow.com/a/38271957
            pool.join()
        except KeyboardInterrupt:
            # Also need to handle Ctrl-C in join()...
            # This might necessitate pressing Ctrl-C again, but it's
            # better than getting spammed by traceback full of garbage.
            pool.terminate()
            was_interrupted = True

        # Start the KeyboardInterrupt trace here.
        if was_interrupted:
            raise KeyboardInterrupt

        return result
