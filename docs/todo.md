Todo
================================================

Minor:
* Rm all occurances of plt.ion(). Works better without.
* Cut back on "config" wording, in favour of xp.
* Rename dpr01.py occurences to some_settings_01.py
* Use unpack_uqs in tabulate_avrgs
* Rm tabulate() in favour of tabulate_orig().
* Replace all np.vectorize in math.py with vectorize0


Major:
* Improve docs
* Make superclasses for the filter, smoother, and iterative smoother.
* Simplify and/or generalize time management?
    * k,kObs only, yielded by ticker
    * Change KObs to KObs-1
    * Note (somewhere) the implicit dependence on t=0 being special
* Simplify and/or improve cov matrix stuff.


Bugs:
* Window focus when plt.ion().
