"""Random number generation."""

import time

import numpy.random as _rnd

rng = _rnd.default_rng()


def set_seed(sd="clock"):
    """Set state of DAPPER random number generator."""
    if (sd is not False) and sd == 0:
        msg = (
            "Seeding with 0 is not a good idea, because\n"
            "- Might be confused with [None, False].\n"
            "- Sometimes people seed experiment k with seed(k*sd),\n"
            "  which is intended to vary with k, but is 0 ∀k."
        )
        raise RuntimeError(msg)

    if sd in [None, "clock"]:
        microsec = int(10**6 * time.time())
        MAXSEED = 2**32
        sd = microsec % MAXSEED

    # Don't set seed if sd==False
    # (but None has already been converted to "clock")
    if sd:
        rng.bit_generator.state = _rnd.default_rng(sd).bit_generator.state

    return rng
