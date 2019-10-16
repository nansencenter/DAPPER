Todo
================================================

* Rename 95 96
* hot, hoteit, sak, sakov
* Localization: use general_local
* Change KObs to KObs-1
* https://stackoverflow.com/q/1024435/how-to-fix-python-indentation
* Replace equi_spaced_integers by linspace_int. Rerun all tests.

* Improve docs
* Make superclasses for the filter, smoother, and iterative smoother.
* Note (somewhere) the implicit dependence on t=0 being special
* Simplify and/or generalize time management?
* Simplify and/or improve cov matrix stuff.

Bugs:
* Window focus.
* Is this why ctrl-c fails so often?
    (from https://docs.python.org/3.5/library/select.html ):
    "Changed in version 3.5:
    The function is now retried with a recomputed timeout when interrupted by a signal,
    except if the signal handler raises an exception (see PEP 475 for the rationale),
    instead of raising InterruptedError."

