"""YAML-based repr mixin for structured objects.

Compared to bare `yaml.dump()`, this module adds:

- **numpy support**: 1-D arrays serialise as inline lists; N-D arrays use
  block literal style (`|`) so they render like `str(arr)` without
  truncating to a single line.
- **numpy scalar support**: `np.int64` etc. are unwrapped to Python
  scalars instead of emitting a `!!python/object/apply` tag.
- **`YamlRepr` mixin**: gives any class a `__repr__` driven by
  `_repr_fields()` — override that method to control which fields appear
  and under what names.
- **nested `YamlRepr` objects**: registered via `add_multi_representer`
  so they inline naturally when nested inside another.
- **unknown-type fallback**: anything without a representer falls back to
  `repr(obj)` as a plain string, rather than raising `RepresenterError`.
"""

import numpy as np
import yaml


class _ReprDumper(yaml.SafeDumper):
    pass


def _np_repr(d, v):
    if v.ndim <= 1:
        return d.represent_data(v.tolist())
    return d.represent_scalar("tag:yaml.org,2002:str", str(v), style="|")


_ReprDumper.add_representer(np.ndarray, _np_repr)
for _t in (np.bool_, np.int32, np.int64, np.float32, np.float64):
    _ReprDumper.add_representer(_t, lambda d, v: d.represent_data(v.item()))


def _yaml_representer(dumper, obj):
    return dumper.represent_data({type(obj).__name__: obj._repr_fields()})


class YamlRepr:
    """Mixin: YAML-formatted __repr__ driven by _repr_fields()."""

    def _repr_fields(self):
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}

    def __repr__(self):
        return yaml.dump(
            {type(self).__name__: self._repr_fields()},
            Dumper=_ReprDumper,
            default_flow_style=None,
            sort_keys=False,
        ).rstrip()


_ReprDumper.add_multi_representer(YamlRepr, _yaml_representer)

# Fallback: anything without a representer → use its __repr__ string
_ReprDumper.add_representer(None, lambda d, v: d.represent_str(repr(v)))


def yaml_repr(obj):
    """Return a compact YAML string for any object."""
    return yaml.dump(
        obj, Dumper=_ReprDumper, default_flow_style=None, sort_keys=False
    ).rstrip()
