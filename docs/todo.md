Todo
================================================
Minor:
* Dont use .ini config files
* Open ipdb session during liveplotting
* Cut back on "config" wording, in favour of xp.
* Rename dpr01.py occurences to some_settings_01.py
* Replace all np.vectorize in math.py with vectorize0


Major:
* Improve docs
* Make superclasses for the filter, smoother, and iterative smoother.
* Simplify and/or generalize time management?
    * k,kObs only, yielded by ticker
    * Change KObs to KObs-1
    * Note (somewhere) the implicit dependence on t=0 being special
* Simplify and/or improve cov matrix stuff.
