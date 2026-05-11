---
name: stats.py modernisation (LSP, typed Avrgs)
description: Work done to modernise dapper/stats.py and series.py for LSP completion and less magic
type: project
---

Completed modernisation of `dapper/stats.py` and `dapper/tools/series.py` on branch `modernize`.

**Key changes:**

- `DACycleSeries` now declares field-summary children (`m`, `ms`, `rms`, `ma`, `gm`) and array subscripts (`f`, `s`) as class-level annotations → enables `xp.stats.err.rms.a` completion.
- Three new typed classes in `stats.py`:
  - `_StatBuffer(DotDict)` — internal scratch dict used in `Stats.assess` (replaces `Avrgs()` there)
  - `DACycleAvrgs(StatPrint, DotDict)` — leaf level, holds `f/a/s/i: UncertainQtty`
  - `FieldAvrgs(StatPrint, DotDict)` — intermediate level, holds `m/ms/rms/ma/gm: DACycleAvrgs`
