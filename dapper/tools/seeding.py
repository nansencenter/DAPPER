"""Random number generation."""

import time

import numpy as np


def set_seed(sd="clock"):
    """Essentially ``np.random.seed(sd)``, but also returning ``sd``.

    The disadvantage of ``np.random.seed()`` is that it produces a
    RandomState that cannot be represented by a simple seed
    (the mapping seed-->state being non-surjective).
    By contrast, when ``sd in [None,"clock"]``, `set_seed` generates
    the seed from the time (in microseconds), which can then be returned.

    If ``sd==False``: do nothing.

    Seed management enables reproducibility of random experiments.
    Moreover, using the same seed for each method in a comparative experiment
    may yield a form of "Variance reduction", eg. CRN, ref Wikipedia.
    This CRN trick is often handy for speeding-up comparisons
    but should not be relied upon in publications,
    which should simply use converged statistics.

    Why is sd=3000 used in many places in DAPPER? Coz I like the number.
    Example use: "André 3000", "I love you 3000".

    Note: why are we using global generator?
    Because that's what we were used to.
    And we're not not worried about thread safety.

    Example:
    >>> def prnt(sd):
    >>>     print(str(sd).ljust(10), np.random.randint(0,99))
    >>> sd = set_seed(3) ; prnt(sd)
    >>> sd = set_seed()  ; prnt(sd)
    >>> sd = set_seed(sd); prnt(sd)
    >>> sd = set_seed(3) ; prnt(sd)
    >>> sd = set_seed()  ; prnt(sd)
    """

    if (sd is not False) and sd == 0:
        msg = ("Seeding with 0 is not a good idea, because\n"
               "- Might be confused with [None, False].\n"
               "- Sometimes people seed experiment k with seed(k*sd),\n"
               "  which is intended to vary with k, but is 0 ∀k.")
        raise RuntimeError(msg)

    if sd in [None, "clock"]:
        microsec = int(10**6 * time.time())
        MAXSEED = 2**32
        sd = microsec % MAXSEED

    if sd:
        np.random.seed(sd)

    return sd
