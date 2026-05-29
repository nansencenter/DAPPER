## Default statistics

For a tabular overview of all default statistics, see the
[Statistics section of the README](https://github.com/nansencenter/DAPPER#statistics).

### `DACycleSeries` subscripts

The time series of statistics (the attributes of `.stats`) are
[`DACycleSeries`][tools.series.DACycleSeries] objects.
They may have sub-arrays `.f`, `.a`, `.s`, `.i`, referring to whether the
statistic is for a "forecast", "analysis", "smoothed" (smoothers only),
or "integrational" (forecast including intermediate, non-obs-time step) estimate,
as determined by the `fais` argument passed to
[Stats.assess][stats.Stats.assess].

The same subscripts apply for the time-averages stored in `.avrgs`.

### Field summaries

The statistics are also averaged in space.
This is done according to the methods listed in
`rc.field_summaries` of the [`dpr_config`][].

!!! note

    Although sometimes pretty close, `rmv` (a.k.a. `spread.rms`) is not (supposed
    to be) an un-biased estimator of `rmse` (a.k.a. `err.rms`).  This is because
    of the square roots involved in the field summary.  Instead, `spread.ms` (i.e.
    the mean variance) is the unbiased estimator of `err.ms`.

### Regional field summaries

If the `HiddenMarkovModel` has the attribute `.sectors` with value, e.g.,

```python
HMM.sectors = {
    "ocean": inds_of_state_of_the_ocean,
    "atmos": inds_of_state_of_the_atmosphere,
}
```

then `.stats.rms` and `.avrgs.rms` will also have attributes
named after the keys of `HMM.sectors`, e.g. `stats.err.rms.ocean`.
This also goes for any other (than `rms`) type of field summary method.

## Declaring new, custom statistics

Only the time series created with [Stats.new_series][stats.Stats.new_series] will be in the format
operated on by [Stats.average_in_time][stats.Stats.average_in_time].  For example, to add a
full DA-cycle series for estimated inflation values:

```python
self.stats.new_series('infl', 1)
```

For a statistic that only exists at analysis times (no `.f` or `.i` sub-arrays):

```python
self.stats.new_series('infl', 1, analysis_only=True)
```

Alternatively you can overwrite a default statistic; for example:

```python
error_time_series_a = xx - ensemble_time_series_a.mean(axis=1)
self.stats.err.rms.a = np.sqrt(np.mean(error_time_series_a**2, axis=-1))
```

To register a scalar (non-series) custom statistic so it appears in `xp.avrgs`,
use `self.stats.register`:

```python
self.stats.register("your_custom_stat", value)
```

This calls [`register`][stats.register] internally, which sets the
attribute and records `"your_custom_stat"` so that
[`xp_launch.run_experiment`][] (which deletes `.stats` after assimilation to
save memory) can still copy it into `xp.avrgs`.
