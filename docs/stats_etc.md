## Default statistics

List them using

>>> list(vars(xp.stats))
... list(vars(xp.avrgs))

### The `FAUSt` key/attribute

The time series of statistics (the attributes of `.stats`) may have attributes
`.f`, `.a`, `.s`, `.u`, referring to whether the statistic is for a "forecast",
"analysis", or "smoothing" estimate (as is decided when the calls to
`Stats.assess` is made), or a "universal" (forecast, but at intermediate
[non-obs.-time]) estimate.

The same applies for the time-averages of `.avrgs`.

### Field summaries

The statistics are also averaged in space.
This is done according to the methods listed in `dpr.rc.field_summaries`.

.. note::
    Although sometimes pretty close, `rmv` (a.k.a. `spread.rms`) is not (supposed
    to be) an un-biased estimator of `rmse` (a.k.a. `err.rms`).  This is because
    of the square roots involved in the field summary.  Instead, `spread.ms` (i.e.
    the mean variance) is the unbiased estimator of `err.ms`.

### Regional field summaries

If the `HiddenMarkovModel` has the attribute `.sectors` with value, e.g.,

>>> HMM.sectors = {
...     "ocean": inds_of_state_of_the_ocean,
...     "atmos": inds_of_state_of_the_atmosphere,
... }

then `.stats.rms` and `.avrgs.rms` will also have attributes
named after the keys of `HMM.sectors`, e.g. `stats.err.rms.ocean`.
This also goes for any other (than `rms`) type of field summary method.

## Declaring new, custom statistics

Only the time series created with `Stats.new_series` will be in the format
operated on by `Stats.average_in_time`.  For example, create `ndarray` of
length `Ko+1` to hold the time series of estimated inflation values:

>>> self.stats.new_series('infl', 1, Ko+1)

Alternatively you can overwrite a default statistic; for example:

>>> error_time_series_a = xx - ensemble_time_series_a.mean(axis=1)
... self.stats.err.rms.a = np.sqrt(np.mean(error_time_series_a**2, axis=-1))

Of course, you could just do this

>>> self.stats.my_custom_stat = value

However, `dapper.xp_launch.run_experiment` (without `free=False`) will delete
the `Stats` object from `xp` after the assimilation, in order to save memory.
Therefore, in order to have `my_custom_stat` be available among `xp.avrgs`, it
must be "registered":

>>> self.stats.stat_register.append("my_custom_stat")

Alternatively, you can do both at once

>>> self.stat("my_custom_stat", value)
