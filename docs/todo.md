Todo
================================================

* do something about fnoise_treatm
* mv liveplotting (kw) liveplots
* abbreviations (aliases) ? Could use __post_init__().
* Relocate ensure_attr, find_1st, simulate, de_abbreviate, tabulate_orig
* Sort out import order

* Rm HMM from Stats
* Make average_stats (and average_in_time) accept indices (or conditions)
    that define regions (for example, "ocean", and "BurnIn")

* Improve docs
* Make superclasses for the filter, smoother, and iterative smoother.
* Simplify and/or generalize time management?
* Simplify and/or improve cov matrix stuff.
* Note (somewhere) the implicit dependence on t=0 being special
* Use pandas for stats time series?
* https://stackoverflow.com/q/1024435/how-to-fix-python-indentation

Bugs:
* Window focus.
* Is this why ctrl-c fails so often?
    (from https://docs.python.org/3.5/library/select.html ):
    "Changed in version 3.5:
    The function is now retried with a recomputed timeout when interrupted by a signal,
    except if the signal handler raises an exception (see PEP 475 for the rationale),
    instead of raising InterruptedError."

