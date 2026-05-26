"""Minimal dict/struct utilities (replaces struct-tools dependency)."""

import itertools
from copy import deepcopy


class DotDict(dict):
    """Dict that also supports attribute (dot) access.

    Example:
    >>> d = DotDict(x=1, y=2)
    >>> d.x
    1
    >>> d.z = 3
    >>> d['z']
    3
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def __deepcopy__(self, memo):
        self2 = self.__class__()
        memo[id(self)] = self2
        for k, v in self.items():
            self2[k] = deepcopy(v, memo)
        return self2


def deep_getattr(obj, name, *default):
    """Chained getattr using dot-separated name.

    Example:
    >>> class A: pass
    >>> a = A(); a.b = A(); a.b.c = 42
    >>> deep_getattr(a, 'b.c')
    42
    """
    for n in name.split("."):
        obj = getattr(obj, n, *default)
    return obj


def deep_hasattr(obj, name):
    """True if deep_getattr succeeds."""
    try:
        deep_getattr(obj, name)
        return True
    except AttributeError:
        return False


def flexcomp(x, *criteria):
    """Compare x against criteria: callable, regex, or value.

    Example:
    >>> flexcomp('foo', 'foo', 'bar')
    True
    >>> flexcomp('baz', 'foo', 'bar')
    False
    """

    def _compare(x, y):
        try:
            return y(x)
        except TypeError:
            pass
        try:
            return bool(y.search(x))
        except AttributeError:
            return y == x

    return any(_compare(x, y) for y in criteria)


def _filter(iterable, criteria, inv=False):
    def negate(x):
        return (not x) if inv else x

    keys = [k for k in iterable if negate(flexcomp(k, *criteria))]
    if isinstance(iterable, dict):
        return {k: iterable[k] for k in keys}
    return keys


def intersect(iterable, wanteds):
    """Keep elements matching any of wanteds (via flexcomp).

    Example:
    >>> intersect({'a': 1, 'b': 2, 'c': 3}, ['a', 'c'])
    {'a': 1, 'c': 3}
    """
    return _filter(iterable, wanteds, inv=False)


def complement(iterable, unwanteds):
    """Keep elements NOT matching any of unwanteds (via flexcomp).

    Example:
    >>> complement({'a': 1, 'b': 2, 'c': 3}, ['b'])
    {'a': 1, 'c': 3}
    """
    return _filter(iterable, unwanteds, inv=True)


def prodct(dct):
    """Cartesian product of dict values, yielding dicts.

    Example:
    >>> list(prodct(dict(n=[1, 2], c='ab')))
    [{'n': 1, 'c': 'a'}, {'n': 1, 'c': 'b'}, {'n': 2, 'c': 'a'}, {'n': 2, 'c': 'b'}]
    """
    return (dict(zip(dct, x)) for x in itertools.product(*dct.values()))


def transps(thing2d, *args, **kwargs):
    """Delegate transpose to the appropriate helper based on container types."""
    dict1 = isinstance(thing2d, dict)
    item0 = next(iter(thing2d.values())) if dict1 else thing2d[0]
    dict2 = isinstance(item0, dict)
    if dict1 and dict2:
        return _transpose_dicts(thing2d, *args, **kwargs)
    elif dict1 and not dict2:
        return _transpose_dict_of_lists(thing2d, *args, **kwargs)
    elif not dict1 and dict2:
        return _transpose_list_of_dicts(thing2d, *args, **kwargs)
    else:
        return _transpose_lists(thing2d, *args, **kwargs)


def _transpose_dicts(DD, safe=True):
    new = {}
    prev = "UNINITIALIZED"
    for i in DD:
        if safe and prev != "UNINITIALIZED":
            assert prev == DD[i].keys(), f"Key mismatch for row {i}"
        prev = DD[i].keys()
        for j in DD[i]:
            new.setdefault(j, {})[i] = DD[i][j]
    return new


def _transpose_list_of_dicts(LD, safe=True):
    new = {}
    prev = "UNINITIALIZED"
    for i, D in enumerate(LD):
        if safe and prev != "UNINITIALIZED":
            assert prev == D.keys(), f"Key mismatch for dict number {i}"
        prev = D.keys()
        for j in D:
            new.setdefault(j, []).append(D[j])
    return new


def _transpose_dict_of_lists(DL, safe=True):
    if safe:
        lens = [len(DL[k]) for k in DL]
        assert all(lens[0] == L for L in lens), "Rows have unequal lengths."
    return [dict(zip(DL, t)) for t in zip(*DL.values())]


def _transpose_lists(LL, safe=True, as_list=False):
    if safe:
        assert all(len(LL[0]) == len(row) for row in LL)
    new = zip(*LL)
    if as_list:
        new = [list(row) for row in new]
    return new
