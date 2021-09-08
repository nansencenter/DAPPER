Only the time series created with `Stats.new_series` will be in the format
operated on by `Stats.average_in_time`.  For example, create `ndarray` of
length `KObs+1` to hold the time series of estimated inflation values:
>>> self.stats.new_series('infl', 1, KObs+1)

Alternatively you can overwrite a default statistic; for example:
>>> error_time_series = xx - ensemble_time_series.mean(axis=1)
... self.stats.err.rms.a = np.sqrt(np.mean(error_time_series**2, axis=-1))

Of course, you could just do this
>>> self.stats.my_custom_stat = value

Moreover, recall that `xp.launch` (without `free=False`) will delete
the `Stats` object from `xp` after the assimilation, in order to save memory.
Therefore, in order to have `my_custom_stat` be available among `xp.avrgs`,
it must be "registered":
>>> self.stats.stat_register.append("my_custom_stat")

Alternatively, you can do both at once
>>> self.stat("my_custom_stat", value)

The statistics are also averaged in space.
This is done according to the field summary methods in listed in `dpr.rc`.

.. note::
    `rmv` (un-abbreviated `std.rms`) is not an un-biased estimator of `rmse`
    (un-abbreviated name `err.rms`).  This is because of the square roots
    involved.  Instead, `std.ms` (i.e. the mean variance) is the unbiased
    estimator of `err.ms`.
