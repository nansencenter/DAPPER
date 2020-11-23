"""Tools for dicts (and lists)."""

# TODO 9
# - Make pip package, answer SO.com questions about
#   https://www.google.com/search?q=python+aligned+dict&oq=python+aligned+dict

import itertools
import shutil
import sys
# Since textwrap() only treats strings, while pprint is python-aware,
# I would have preferred pprint.pformat(width=lw,compact=True,sort_dicts=False),
# but pprint refuses to use my repr.
# It also could mess up my recursion guard, which inserts "<Recursion...>",
# because it tests for x.startswith("<").
import textwrap
from _thread import get_ident
from contextlib import contextmanager

import numpy as np


def get0(dct):
    """Get first value."""
    # Bad?: dct[list(dct.keys())[0]]
    return next(iter(dct.values()))


# TODO 4: rm?
def flexcomp(x, *criteria):
    """Compare in various ways."""

    def _compare(x, y):
        # Callable (condition) compare
        try:
            return y(x)
        except TypeError:
            pass
        # Regex compare -- should be compiled on the outside
        try:
            return bool(y.search(x))
        except AttributeError:
            # Value compare
            return y == x

    return any(_compare(x, y) for y in criteria)


def _intersect(iterable, criteria, inv=False):
    """Keep elements of ``iterable`` that match **any** criteria,

    as evaluated by flexcomp().

    Returns dict/list if ``iterable`` is dict/iterable."""
    def negate(x): return (not x) if inv else x
    keys = [k for k in iterable if negate(flexcomp(k, *criteria))]
    if isinstance(iterable, dict):  # Dict
        return {k: iterable[k] for k in keys}
    return keys


# For some reason, _intersect (with the `inv` switch) is
# for the brain to parse. Use intersect and complement instead.
def intersect(iterable, wanteds):
    return _intersect(iterable, wanteds, inv=False)


def complement(iterable, unwanteds):
    return _intersect(iterable, unwanteds, inv=True)


# Complement should be called "relative complement" (or set diff)


def prodct(dct):
    """Cartesian/Outer product, for dicts.

    Example:
    >>> list(prodct(dict(n=[1,2], c='ab')))
    [{'n': 1, 'c': 'a'},
     {'n': 1, 'c': 'b'},
     {'n': 2, 'c': 'a'},
     {'n': 2, 'c': 'b'}]

    Source: https://stackoverflow.com/a/40623158/38281."""
    return (dict(zip(dct, x)) for x in itertools.product(*dct.values()))


def transpose_dicts(DD, safe=True):
    """As the name says.

    Incredibly, it is difficult to make this less verbose
    (the one-liner is unpredicable for non-rectangular cases)

    Example:
    >>> DD = {chr(97+i): {chr(65+j):i*10+j for j in range(2)}
                                           for i in range(3)}
    >>> DD
    {'a': {'A': 0, 'B': 1}, 'b': {'A': 10, 'B': 11}, 'c': {'A': 20, 'B': 21}}

    >>> transpose_dicts(DD)
    {'A': {'a': 0, 'b': 10, 'c': 20}, 'B': {'a': 1, 'b': 11, 'c': 21}}
    """
    new = {}
    prev = "UNINITIALIZED"
    for i in DD:

        # Validate column keys
        if safe and prev != "UNINITIALIZED":
            assert prev == DD[i].keys(), f"Key mismatch for row {i}"
        prev = DD[i].keys()

        for j in DD[i]:
            new.setdefault(j, {})[i] = DD[i][j]

    return new


def transps(thing2d, *args, **kwargs):
    """Delegates transpose operation to the appropriate function."""

    # Is dict? Otherwise assume list (tuple also works).
    dict1 = isinstance(thing2d, dict)
    item0 = get0(thing2d) if dict1 else thing2d[0]  # fine since py3.7
    dict2 = isinstance(item0, dict)

    if       dict1 and     dict2: f = transpose_dicts           # noqa
    elif     dict1 and not dict2: f = transpose_dict_of_lists   # noqa
    elif not dict1 and     dict2: f = transpose_list_of_dicts   # noqa
    elif not dict1 and not dict2: f = transpose_lists           # noqa

    return f(thing2d, *args, **kwargs)


def transpose_list_of_dicts(LD, safe=True):
    """

    Example:
    >>> LD = [{chr(97+j):j for j in range(2)} for i in range(3)]
    >>> LD
    [{'a': 0, 'b': 1}, {'a': 0, 'b': 1}, {'a': 0, 'b': 1}]
    >>> transpose_list_of_dicts(LD)
    {'a': [0, 0, 0], 'b': [1, 1, 1]}
    """
    new = {}
    prev = "UNINITIALIZED"
    for i, D in enumerate(LD):

        # Validate column keys
        if safe and prev != "UNINITIALIZED":
            assert prev == D.keys(), f"Key mismatch for dict number {i}"
        prev = D.keys()

        for j in D:
            new.setdefault(j, []).append(D[j])

    return new


def transpose_dict_of_lists(DL, safe=True):
    """This is essentially a one-liner.

    However, the validation of `safe=True` requires some effort.

    Example:
    >>> DL = {chr(97+i): [i*10+j for j in range(2)] for i in range(3)}
    >>> DL
    {'a': [0, 1], 'b': [10, 11], 'c': [20, 21]}
    >>> transpose_dict_of_lists(DL)
    [{'a': 0, 'b': 10, 'c': 20}, {'a': 1, 'b': 11, 'c': 21}]
    """

    # Validate row length
    if safe:
        lens = [len(DL[k]) for k in DL]
        assert all(lens[0] == L for L in lens), "Rows have unqual lengths."

    # https://stackoverflow.com/q/5558418
    new = [dict(zip(DL, t)) for t in zip(*DL.values())]

    return new


def transpose_lists(LL, safe=True, as_list=False):
    """This is essentially `zip(*LL)`.

    However, the validation of `safe=True` requires some effort,
    and converting to an explicit list requires another bit.

    Example:
    >>> LL = [[i*10 + j for j in range(2)] for i in range(3)]
    >>> LL
    [[0, 1], [10, 11], [20, 21]]
    >>> transpose_lists(LL,as_list=True)
    [[0, 10, 20], [1, 11, 21]]
    """
    if safe:
        assert all(len(LL[0]) == len(row) for row in LL)

    new = zip(*LL)  # transpose

    if as_list:
        # new = list(map(list, new))
        new = [list(row) for row in new]

    return new


# from https://pingfive.typepad.com/blog/2010/04/
def deep_getattr(obj, name, *default):
    for n in name.split("."):
        obj = getattr(obj, n, *default)
    return obj


# Rm'd coz it's what type of obj to set at intermediate hierarchy levels:
# deep_setattr()


def deep_hasattr(obj, name):
    try:
        deep_getattr(obj, name)
        return True
    except AttributeError:
        return False


MIN_LINEWIDTH = 5

# TODO 4: should also be made thread-safe? Ref _is_being_printed.
_linewidth = None
_top_dog = None
_is_being_printed = set()


@contextmanager
def _shorten_linewidth_by(n):
    """Shorten linewidth parameter. Also works on numpy."""

    # Avoid (overhead of) importing numpy if not required.
    # https://stackoverflow.com/a/30483269/38281
    if "numpy" in sys.modules:
        def np_lw(lw): return np.set_printoptions(linewidth=lw)
    else:  # Provide pass-through
        def np_lw(lw): return lw

    # Structured (store, try, finally) just like np.printoptions().
    global _linewidth
    old = _linewidth
    try:
        # Set lw
        _linewidth -= n
        _linewidth = max(_linewidth, MIN_LINEWIDTH)
        np_lw(_linewidth)
        yield _linewidth
    finally:
        # Restore lw
        _linewidth = old
        np_lw(old)


def _init(user_function):
    """Does two things:

    - Monitor recursion of self.

      - This is done is just like in ``reprlib``
        whence I learned how to make it thread-safe.
        One diff. is that here _repr_running is global,
        which works better for this convoluted example:
        >>> d1 = AlignedDict(a=1)
        >>> d1["d2"] = AlignedDict(self=d1)
        >>> d1["lst"] = [d1["d2"]]

      - Sibling repetition is not recursion.

    - Initializes the linewidth with ``get_terminal_size()``,
      unless the ``_linewidth`` variable has been set.

    Note: Both of these actions also require clean-up,
    so there's a ``finally`` section as well.

    Note: Cannot apply to core printing function (AlignedDict._repr),
    because each call to AlignedDict creates new (unregistered) id's."""

    def wrapped(self):

        # Set linewidth
        global _top_dog, _linewidth
        old_linewidth = _linewidth
        if _top_dog is None:
            _top_dog = id(self)

            if _linewidth is None:
                _linewidth, _ = shutil.get_terminal_size()

        # Turn ON recursion guard
        id_thread = id(self), get_ident()
        if id_thread in _is_being_printed:
            return "<Recurs-%d>" % id(self)
        # return "..."
        _is_being_printed.add(id_thread)

        try:
            result = user_function(self)

        finally:
            # Turn OFF recursion guard
            _is_being_printed.discard(id_thread)

            # Restore linewidth
            if id(self) == _top_dog:
                _top_dog = None
                _linewidth = old_linewidth

        return result

    return wrapped


class AlignedDict(dict):
    """Dict whose items are printed aligned and line-wrapped.

    Initialized and used just like a regular dict, since only
    difference to ``dict`` is __str__ and __repr__ (of which
    the other functions are auxiliaries and hidden).

    Also provides printopts. Example:

    >>> a = np.arange(24)
    >>> A = a.reshape((4,-1))
    >>> dct = AlignedDict(a=a, b=99, z=3, A=A)
    >>> dct["self"] = dct
    >>> dct["sub"] = AlignedDict(x=1,text="lorem",self=dct)
    >>> dct.printopts = dict(excluded=["z"], aliases={"A":"matrix"})
    >>> print(repr(dct))

    Note on ``printopts["indent"]``:
    - If not set,
      then the keys are right-aligned (so that the colons line up),
      and the values (including multi-line) are printed to their right,
      allowing them to be placed on the same line as the key.
      This setting is pretty, unless the keys are really long.
    - If set (to an integer), then the values are printed on a new line,
      indented by that (unchanging) length (unless they're one-liners).

    A similar class is scipy.optimize.OptimizeResult

    See also:
    - pprint.pformat(x,linewidth,compact=True,sort_dicts=False)
    - json.dumps(x, indent=4, sort_keys=False, default=repr)
    But, both of these work recursively, which is NOT our aim.
    In particular, they need to implement behaviour for all types.
    """

    @_init
    def __str__(self):
        return self._repr(is_repr=False)

    @_init
    def __repr__(self):
        return self._repr(is_repr=True)

    def _repr(self, is_repr=False):
        """Both __str__ and __repr__ in one!

        Note: if ``self`` is printed as part of a standard object,
        __repr__ gets called, not __str__."""

        # Constants
        sep = ": "
        bullet = " " if is_repr else "  - "
        comma = "," if is_repr else ""
        spinechar = " " if is_repr else "¦"  # ¦, ┆, │, ⎸, ▏
        format_key = repr if is_repr else str
        format_val = repr if is_repr else str

        # Apply print options
        dct = self
        opt = getattr(self, "printopts", {})
        dct = intersect(dct,  opt.get("included", dct))
        dct = complement(dct, opt.get("excluded", []))
        dct = {opt.get("aliases", {}).get(k, k): v for k, v in dct.items()}

        if "indent" in opt:
            # LEFT-aligned keys -- indent is fixed, and val is always on newline
            val_indent = " " * len(bullet) + spinechar.ljust(opt["indent"])
        else:
            # RIGHT-aligned keys -- indent ~ on keys, and val can start on key's line
            kWidth = max([len(format_key(k)) for k in dct], default=0)
            val_indent = " " * len(bullet) + " " * kWidth + spinechar.ljust(len(sep))

            def format_key(key, old=format_key):
                return old(key).rjust(kWidth)

        def iRepr(key, val):
            trim = len(val_indent + comma)

            # Try placing on 1 line, but use max() to insure against wrapping
            if "indent" in opt and ("\n" not in format_val(val)):
                trim = max(trim, len(bullet + format_key(key) + sep + comma))

            with _shorten_linewidth_by(trim):
                val = self._wrap_item(format_val(val))

            val = ("\n" + val_indent).join(val)

            if "indent" in opt and ("\n" in val):
                val = "\n" + val_indent + val  # 1st line also on new line

            return format_key(key) + sep + val

        # dct --> text
        items = [iRepr(k, v) for k, v in dct.items()]

        # Sort
        items = self._sort(zip(dct, items))

        # Join
        txt = (comma + "\n" + bullet).join(items)
        if len(items) > 1:
            txt = "\n" + bullet + txt + "\n"

        # Add braces, etc
        if is_repr:
            txt = "{" + txt + "}"

        return txt

    # AUX FUNCTIONS
    @staticmethod
    def _wrap_item(txt):
        """Wrap txt."""
        lines = txt.splitlines()
        ll = [textwrap.wrap(line, _linewidth) for line in lines]  # list of lists
        hang = " "  # No need to shorten_lw for this
        ll = [w[:1] + [hang + line for line in w[1:]] for w in ll]
        ll = [k for w in ll for k in w]  # flatten list-of-lists
        return ll

    def _sort(self, dict_and_texts):
        ORDR = getattr(self, "printopts", {}).get("ordering", []) or []
        reverse = getattr(self, "printopts", {}).get("reverse", False)

        def indexer(key_text):
            key, text = key_text
            if not isinstance(ORDR, str):  # assume list
                # Manual ordering.
                if key in ORDR:
                    # -10000 => priority
                    idx = -10000 + ORDR.index(key)
                else:
                    idx = 0  # neutral
            elif "line" in ORDR:
                # Line-number count
                idx = 100 * text.count("\n")
                # Line-length count
                idx += max(len(line) for line in text.splitlines())
            elif "alpha" in ORDR:
                # Alphabetic (tuple compare)
                idx = key.lower()
            return idx

        pairs = sorted(dict_and_texts, key=indexer, reverse=reverse)
        return [text for key, text in pairs]


class NicePrint:
    """Provides __repr__ and __str__ by AlignedDict(vars(self)).

    Example usage:
    >>> class MyClass(NicePrint):
    >>>     printopts = NicePrint.printopts.copy()
    >>>     printopts["excluded"] += ["my_hidden_var"]
    >>>     printopts["aliases"] = {"asdf":"better_name"}
    >>>     ...
    """

    # _underscored = re.compile('^_')
    _underscored = lambda s: s.startswith("_") # noqa
    printopts = dict(excluded=[_underscored, "printopts"])

    def _repr(self, is_repr=True):
        cls_name = type(self).__name__ + "(" if is_repr else ""
        dct = AlignedDict(vars(self))
        dct.printopts = self.printopts
        with _shorten_linewidth_by(len(cls_name)):
            txt = repr(dct) if is_repr else str(dct)
            txt = ("\n" + " " * len(cls_name)).join(txt.splitlines())
        return cls_name + txt + (")" if is_repr else "")

    @_init
    def __str__(self):
        return self._repr(is_repr=False)

    @_init
    def __repr__(self):
        return self._repr(is_repr=True)


class DotDict(AlignedDict):
    """Dict that *also* supports attribute (dot) access.

    Benefit compared to a dict:

     - Verbosity of ``d['a']`` vs. ``d.a``.
     - Includes ``AlignedDict``.

    DotDict is not terribly hackey, and is quite robust.
    Similar constructs are quite common, eg IPython/utils/ipstruct.py.

    Main inspiration: https://stackoverflow.com/a/14620633
    """

    printopts = dict(excluded=["printopts"])

    def __init__(self, *args, **kwargs):
        """Init like a normal dict."""
        super().__init__(*args, **kwargs)  # Make a (normal) dict
        self.__dict__ = self  # Assign it to self.__dict__


# def print_nested(dct):
#     """Print nested dicts"""
#
#     # Note: if a dict is inside of a list (for example),
#     # it will be printed in the standard fashion,
#     # despite our overriding the builtins.repr.
#     # There appears to be no way around this, except,
#     # like reprlib does, defining custom treatment for each
#     # type (dicts, list, sets, tuples, deques, etc).
#
#     # Another issue is that I dont know how to check if an object
#     # is suitable for AlignedDict printing or not,
#     # so this function is restricted to dicts.
#
#     @_init
#     def new_repr(obj):
#         if hasattr(obj,"items"):
#             obj = AlignedDict(obj)
#         return orig_repr(obj)
#
#     import builtins
#     orig_repr = builtins.repr
#     try:
#         builtins.repr = new_repr
#         txt = repr(dct)
#     finally:
#         builtins.repr = orig_repr
#
#     print(txt)


if __name__ == "__main__":
    # Note: this setup is purposely messy,
    # in order to test recursion treatments.

    a = np.arange(24)
    A = a.reshape((4, -1))

    d1 = dict(a=a, b=99, z=3, A=A)
    d2 = dict(x=1, lorem="ipsum")

    # Dont move this block below (coz then it will contain
    # d1/d2 recursions, rather than a1/a2)
    a1 = AlignedDict(d1)
    a2 = AlignedDict(d2)

    # Json -- cannot handle recursions
    # import json

    d1["d2"] = d2
    # print("\njson.dumps:\n================")
    # print(json.dumps(d1, indent=4, default=repr))

    # pprint
    # import pprint

    d1["d2"] = d2
    d2["d1"] = d1
    d1["lst"] = [0, 1, d2]
    # print("\npprint:\n================")
    # pprint.pprint(d1,compact=True)

    # Regular dict/print
    # print("\nRegular dict/print:\n================")
    # print(d1)

    # Add recursions similar to d1/d2
    # print("\nAlignedDict:\n================")
    a2["a1"] = a1
    a1["a2"] = a2
    a1["one"] = AlignedDict(item="hello")
    a1["empty"] = AlignedDict()
    a1["really long name that goes on and on"] = [0, 1, a2]

    a1.printopts = dict(
        excluded=["z"],
        aliases={"A": "aMatrix"},
        ordering="line",  # or alpha or ["a2", "self"]
        # reverse=True,
    )
    print("\nstr:\n================")
    print(a1)
    print("\nrepr:\n================")
    print(repr(a1))

    print("\n================\nwith const. indent:\n================")

    a1.printopts["indent"] = 1
    a2.printopts = dict(indent=1)
    print("\nstr:\n================")
    print(a1)
    print("\nrepr:\n================")
    print(repr(a1))

    print("\nNicePrint:\n================")

    class MyClass(NicePrint):
        printopts = NicePrint.printopts.copy()

        def __init__(self):
            self._a = 99
            self.a = np.arange(24)
            # self.a = 1
            # self.lorem = "ipsum"
            # self.lst = np.arange(20)

    obj1 = MyClass()
    # obj1.obj2 = MyClass()
    # obj1.self = obj1
    obj1.printopts["excluded"] += ["lst"]
    print(repr(obj1))
    print(obj1)

    print("\nDotDict:\n================")
    dd = DotDict(a=a, b=99, z=3, A=A)
    dd.dd2 = DotDict(a=a, b=99, z=3, A=A)
    dd.self = dd
    dd.printopts["excluded"] += ["A"]
    print(repr(dd))
    print(dd)

    # Other tests
    print("\ndeep_getattr:\n================")
    key2 = "self.self.self.self.a"
    print(key2, ":", deep_getattr(dd, key2))
