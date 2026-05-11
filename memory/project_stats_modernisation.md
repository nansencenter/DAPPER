---
name: stats.py modernisation (LSP, typed Avrgs)
description: Work done to modernise dapper/stats.py and series.py for LSP completion and less magic
type: project
---

Completed modernisation of `dapper/stats.py` and `dapper/tools/series.py` on branch `modernize`.

**Key changes:**

- Added `from __future__ import annotations` to `series.py` and `stats.py` to allow self-referential type hints.
- `DACycleSeries` now declares field-summary children (`m`, `ms`, `rms`, `ma`, `gm`) and array subscripts (`f`, `s`) as class-level annotations → enables `xp.stats.err.rms.a` completion.
- Three new typed classes in `stats.py`:
  - `_StatBuffer(DotDict)` — internal scratch dict used in `Stats.assess` (replaces `Avrgs()` there)
  - `StatAvrg(StatPrint, DotDict)` — leaf level, holds `f/a/s/i: UncertainQtty`
  - `FieldStatAvrg(StatPrint, DotDict)` — intermediate level, holds `m/ms/rms/ma/gm: StatAvrg`
- `Stats` class has typed annotations for all known stat names.
- `Avrgs` now declares typed attributes for LSP; `__getattribute__` magic replaced by `__getattr__` (fires only for missing attrs) + `@property` aliases `rmse`, `rmss`, `rmv`.
- `Stats.average_in_time` rewritten: unified `"fasi"` loop with `getattr(..., None)` + `np.any(np.isfinite(vals))` check (no more separate `store_i` branch or `hasattr` fragility).
- `store_s` check uses `hasattr` instead of `key in xp.__dict__`.

**Why:** All 353 tests pass. `xp.avrgs.err.rms` is now a `StatAvrg`; `xp.avrgs.err.rms.a` is a `UncertainQtty`. LSP completion works end-to-end.
