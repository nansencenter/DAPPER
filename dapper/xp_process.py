"""Tools (notably `xpSpace`) for processing and presenting experiment data."""

import collections
import copy
import logging
import os
import shutil
import warnings
from datetime import datetime
from pathlib import Path

import colorama
import dill
import matplotlib as mpl
import numpy as np
import struct_tools
from matplotlib import cm, ticker
from mpl_tools import place
from patlib.std import nonchalance, set_tmp
from struct_tools import DotDict
from tabulate import tabulate
from tqdm.auto import tqdm

import dapper.tools.remote.uplink as uplink
from dapper.dpr_config import rc
from dapper.stats import align_col, unpack_uqs
from dapper.tools.colors import color_text, stripe
from dapper.tools.rounding import UncertainQtty
from dapper.tools.viz import axis_scale_by_array
from dapper.xp_launch import XP_TIMESTAMP_TEMPLATE, collapse_str, xpList

mpl_logger = logging.getLogger('matplotlib')


class NoneDict(DotDict):
    """DotDict with getattr fallback (None)."""

    def __getattr__(self, name):
        return None


NO_KEY = ("da_method", "xSect", "upd_a")
def make_label(coord, no_key=NO_KEY, exclude=()):  # noqa
    """Make label from coord."""
    dct = {a: v for a, v in coord.items() if v != None}
    lbl = ''
    for k, v in dct.items():
        if k not in exclude:
            if any(x in k for x in no_key):
                lbl = lbl + f' {v}'
            else:
                lbl = lbl + f' {collapse_str(k,7)}:{v}'
    return lbl[1:]


def default_styles(coord, baseline_legends=False):
    """Quick and dirty (but somewhat robust) styling."""
    style = DotDict(ms=8)
    style.label = make_label(coord)

    if coord.da_method == "Climatology":
        style.ls = ":"
        style.c = "k"
        if not baseline_legends:
            style.label = None

    elif coord.da_method == "OptInterp":
        style.ls = ":"
        style.c = .7*np.ones(3)
        style.label = "Opt. Interp."
        if not baseline_legends:
            style.label = None

    elif coord.da_method == "Var3D":
        style.ls = ":"
        style.c = .5*np.ones(3)
        style.label = "3D-Var"
        if not baseline_legends:
            style.label = None

    elif coord.da_method == "EnKF":
        style.marker = "*"
        style.c = "C1"

    elif coord.da_method == "PartFilt":
        style.marker = "X"
        style.c = "C2"

    else:
        style.marker = "."

    return style


def rel_index(elem, lst, default=None):
    """`lst.index(elem) / len(lst)` with fallback."""
    try:
        return lst.index(elem) / len(lst)
    except ValueError:
        if default == None:
            raise
        return default


def discretize_cmap(cmap, N, val0=0, val1=1, name=None):
    """Discretize `cmap` so that it partitions `[0, 1]` into `N` segments.

    I.e. `cmap(k/N) == cmap(k/N + eps)`.

    Also provide the ScalarMappable `sm`
    that maps range(N) to the segment centers,
    as will be reflected by `cb = fig.colorbar(sm)`.
    You can then re-label the ticks using
    `cb.set_ticks(np.arange(N)); cb.set_ticklabels(["A","B","C",...])`.
    """
    # cmap(k/N)
    from_list = mpl.colors.LinearSegmentedColormap.from_list
    colors = cmap(np.linspace(val0, val1, N))
    cmap = from_list(name, colors, N)
    # sm
    cNorm = mpl.colors.Normalize(-.5, -.5+N)
    sm = mpl.cm.ScalarMappable(cNorm, cmap)
    return cmap, sm


def cm_bond(cmap, xp_dict, axis, vmin=0, vmax=0):
    """Map cmap for `coord.axis ∈ [0, len(ticks)]`."""
    def link(coord):
        """Essentially: `cmap(ticks.index(coord.axis))`"""
        if hasattr(coord, axis):
            ticks = xp_dict.ticks[axis]
            cNorm = mpl.colors.Normalize(vmin, vmax + len(ticks))
            ScMap = cm.ScalarMappable(cNorm, cmap).to_rgba
            index = ticks.index(getattr(coord, axis))
            return ScMap(index)
        else:
            return cmap(0.5)
    return link


def in_idx(coord, indices, xp_dict, axis):
    """Essentially: `coord.axis in ticks[indices]`."""
    if hasattr(coord, axis):
        ticks = np.array(xp_dict.ticks[axis])[indices]
        return getattr(coord, axis) in ticks
    else:
        return True


def find_latest_run(root: Path):
    """Find the latest experiment (dir containing many)"""
    def parse(d):
        try:
            return datetime.strptime(d.name, XP_TIMESTAMP_TEMPLATE)
        except ValueError:
            return None
    dd = [e for e in (parse(d) for d in root.iterdir()) if e is not None]
    d = max(dd)
    d = datetime.strftime(d, XP_TIMESTAMP_TEMPLATE)
    return d


def load_HMM(save_as):
    save_as = Path(save_as).expanduser()
    HMM = dill.load(open(save_as/"xp.com", "rb"))["HMM"]
    return HMM


def load_xps(save_as):
    """Load `xps` (as a simple list) from dir."""
    save_as = Path(save_as).expanduser()
    files = [d/"xp" for d in uplink.list_job_dirs(save_as)]

    def load_any(filepath):
        """Load any/all `xp's` from `filepath`."""
        with open(filepath, "rb") as F:
            # If experiment crashed, then xp will be empty
            try:
                data = dill.load(F)
            except EOFError:
                return []
            # Always return list
            try:
                return data["xps"]
            except KeyError:
                return [data["xp"]]

    print("Loading %d files from %s" % (len(files), save_as))
    xps = []  # NB: progbar wont clean up properly w/ list compr.
    for f in tqdm(files, desc="Loading"):
        xps.extend(load_any(f))

    if len(xps) < len(files):
        print(f"{len(files)-len(xps)} files could not be loaded.")

    return xps


def save_xps(xps, save_as, nDir=100):
    """Split xps and save in save_as/i for i in range(nDir).

    Example
    -------
    Rename attr n_iter to nIter:
    >>> proj_name = "Stein"
    >>> dd = rc.dirs.data / proj_name
    >>> save_as = dd / "run_2020-09-22__19:36:13"

    >>> for save_as in dd.iterdir():  # doctest: +SKIP
    ...     save_as = dd / save_as
    ...
    ...     xps = load_xps(save_as)
    ...     HMM = load_HMM(save_as)
    ...
    ...     for xp in xps:
    ...         if hasattr(xp,"n_iter"):
    ...             xp.nIter = xp.n_iter
    ...             del xp.n_iter
    ...
    ...     overwrite_xps(xps, save_as)
    """
    save_as = Path(save_as).expanduser()
    save_as.mkdir(parents=False, exist_ok=False)

    splitting = np.array_split(xps, nDir)
    for i, sub_xps in enumerate(tqdm(splitting, desc="Saving")):
        if len(sub_xps):
            iDir = save_as / str(i)
            os.mkdir(iDir)
            with open(iDir/"xp", "wb") as F:
                dill.dump({'xps': sub_xps}, F)


def overwrite_xps(xps, save_as, nDir=100):
    """Save xps in save_as, but safely (by first saving to tmp)."""
    save_xps(xps, save_as/"tmp", nDir)

    # Delete
    for d in tqdm(uplink.list_job_dirs(save_as),
                  desc="Deleting old"):
        shutil.rmtree(d)

    # Mv up from tmp/ -- goes quick, coz there are not many.
    for d in os.listdir(save_as/"tmp"):
        shutil.move(save_as/"tmp"/d, save_as/d)

    shutil.rmtree(save_as/"tmp")


def reduce_inodes(save_as, nDir=100):
    """Reduce the number of `xp` dirs

    by packing multiple `xp`s into lists (`xps`).

    This reduces the **number** of files (inodes) on the system,
    which limits storage capacity (along with **size**).

    It also deletes files "xp.var" and "out"
    (which tends to be relatively large coz of the progbar).
    This is probably also the reason that the loading time is sometimes reduced.
    """
    overwrite_xps(load_xps(save_as), save_as, nDir)


class SparseSpace(dict):
    """Subclass of `dict` that enforces key conformity to a given `namedtuple`.

    Like a normal `dict`, it can hold any type of objects.
    But, since the keys must conform, they effectively follow a coordinate system,
    so that the `dict` becomes a vector **space**.

    The coordinate system is specified by the `axes`:
    a list of keys defining the `namedtuple` of `self.Coord`.

    In intended usage, this space is highly sparse,
    meaning there are many coordinates with no entry.
    Indeed, as a data format for nd-arrays, it may be called
    "coordinate list representation", used e.g. by `scipy.sparse.coo_matrix`.

    Thus, operations across (potentially multiple) axes,
    such as optimization or averaging, should be carried out by iterating
    -- not over the axes -- but over the the list of items.

    The most important method is `nest`,
    which is used (by `xpSpace.table_tree`) to print and plot results.

    In addition, `__getitem__` is very flexible, allowing accessing by:

    - The actual key, a `self.Coord` object. Returns single item.
    - A `dict` to match against (part of) the coordinates. Returns subspace.
    - An `int`. Returns `list(self)[key]`.
    - A list of any of the above. Returns list.

    This flexibility can cause bugs, but it's probably still worth it.
    Also see `__call__`, `get_for`, and `coords_matching`, for further convenience.

    Inspired by

    - https://stackoverflow.com/a/7728830
    - https://stackoverflow.com/q/3387691

    Example:
    >>> dct = xpSpace(["x", "y", "z"])
    >>> dct[(1, 2, 3)] = "point 1"
    >>> dct[1, 2, 3] == dct[(1, 2, 3)] == dct[dct.Coord(1, 2, 3)] == "point 1"
    True

    This dict only has three dimensions/axes, so this fails:
    >>> dct[(1, 2, 3, 4)]
    Traceback (most recent call last):
    ...
    KeyError: (1, 2, 3, 4)

    Individual coordinates can be anything. For example `None`:
    >>> dct[(1, 2, None)] = "point 2"
    """

    @property
    def axes(self):
        return self.Coord._fields

    def __init__(self, axes):
        """Usually initialized through `xpSpace`.

        Parameters
        ----------
        axes: list
            The attributes defining the coordinate system.
        """
        # Define coordinate system
        self.Coord = collections.namedtuple('Coord', axes)

        # Dont print keys in str
        self.Coord.__str__  = lambda c: "(" + ", ".join(str(v) for v in c) + ")"
        # Only show ... of Coord(...)
        self.Coord.repr2 = lambda c: repr(c).replace("Coord", "").strip("()")

    def update(self, items):
        """Update dict, using the custom `__setitem__` to ensure key conformity.

        NB: the `kwargs` syntax is not supported because it only works for keys that
        consist of (a single) string, which is not very interesting for SparseSpace.
        """
        # See https://stackoverflow.com/a/2588648
        # and https://stackoverflow.com/a/2390997
        try:
            items = items.items()
        except AttributeError:
            pass
        for k, v in items:
            self[k] = v

    def __setitem__(self, key, val):
        """Setitem ensuring coordinate conforms."""
        try:
            key = self.Coord(*key)
        except TypeError:
            raise TypeError(
                f"The key {key!r} did not fit the coord. system "
                f"which has axes {self.axes}")
        super().__setitem__(key, val)

    def __getitem__(self, key):
        """Flexible indexing."""
        # List of items (by a list of indices).
        # Also see get_for().
        if isinstance(key, list):
            return [self[k] for k in key]

        # Single (by integer) or list (by Slice)
        # Note: NOT validating np.int64 here catches quite a few bugs.
        elif isinstance(key, int) or isinstance(key, slice):
            return [*self.values()][key]

        # Subspace (by dict, ie. an informal, partial coordinate)
        elif isinstance(key, dict):
            outer = self.nest(outer_axes=list(key))  # nest
            coord = outer.Coord(*key.values())       # create coord
            inner = outer[coord]                     # chose subspace
            return inner

        # Single item (by Coord object, coz an integer (eg)
        # gets interpreted (above) as a list index)
        else:
            # NB: Dont't use isinstance(key, self.Coord)
            # coz it fails when the namedtuple (Coord) has been
            # instantiated in different places (but with equal params).
            # Also see bugs.python.org/issue7796
            return super().__getitem__(key)

    def __call__(self, **kwargs):
        """Convenient syntax to get/access items.

        Example
        -------
        >>> xp_dict(da_method="EnKF", infl=1, seed=3)  # doctest: +SKIP
        """
        return self.__getitem__(kwargs)

    def get_for(self, ticks, default=None):
        """Almost `[self.get(Coord(x)) for x in ticks]`.

        NB: using the "naive" thing: `[self[x] for x in ticks]`
        would probably be a BUG coz integer `x` gets interpreted as indices
        for the internal list.
        """
        singleton = not hasattr(ticks[0], "__iter__")
        def coord(xyz): return self.Coord(xyz if singleton else xyz)
        return [self.get(coord(x), default) for x in ticks]

    def coord_from_attrs(self, entry):
        """Form a `coord` for this `xpSpace` by extracting attrs. from `obj`.

        **If** the entries of `self` have attributes matching their `coord`s,
        then this can be seen as the inverse of `__getitem__`. I.e.

            self.coord_from_attrs(self[coord]) == coord
        """
        coord = (getattr(entry, a, None) for a in self.axes)
        return self.Coord(*coord)

    def coords_matching(self, **kwargs):
        """Get all `coord`s matching kwargs.

        Unlike `__getitem__(**kwargs)`,

        - A list is returned, not a subspace.
        - This list constains keys (coords), not values.
        - The coords refer to the original space, not the subspace.

        The last point is especially useful for `SparseSpace.label_xSection`.
        """
        def embed(coord):
            return {**kwargs, **coord._asdict()}
        return [self.Coord(**embed(x)) for x in self[kwargs]]

        # Old implementation.
        # - I prefer the new version for its re-use of __getitem__'s
        #   nesting, evidencing their mutual relationship)
        # - Note that unlike xpList.inds(): missingval shenanigans
        #   are here unnecessary coz each coordinate is complete.
        # match  = lambda x: all(getattr(x,k)==kwargs[k] for k in kwargs)
        # return [x for x in self if match(x)]

    def __repr__(self):
        txt  = f"<{self.__class__.__name__}>"
        txt += " with Coord/axes: "
        try:
            txt += "(and ticks): " + str(struct_tools.AlignedDict(self.ticks))
        except AttributeError:
            txt += str(self.axes) + "\n"

        # Note: print(xpList(self)) produces a more human-readable table,
        # but requires prep_table(), which we don't really want to call again
        # (it's only called in from_list, not (necessarily) in any nested spaces)
        L = 2
        keys = [str(k) for k in self]
        if 2*L < len(keys):
            keys = keys[:L] + ["..."] + keys[-L:]
        keys = "[\n  " + ",\n  ".join(keys) + "\n]"
        return txt + f"populated by {len(self)} keys: {keys}"

    def nest(self, inner_axes=None, outer_axes=None):
        """Project along `inner_acces` to yield a new `xpSpace` with axes `outer_axes`

        The entries of this `xpSpace` are themselves `xpSpace`s, with axes `inner_axes`,
        each one regrouping the entries with the same (projected) coordinate.

        Note: is also called by `__getitem__(key)` if `key` is dict.
        """
        # Default: a singleton outer space,
        # with everything contained in the inner (projection) space.
        if inner_axes is None and outer_axes is None:
            outer_axes = ()

        # Validate axes
        if inner_axes is None:
            assert outer_axes is not None
            inner_axes = struct_tools.complement(self.axes, outer_axes)
        else:
            assert outer_axes is None
            outer_axes = struct_tools.complement(self.axes, inner_axes)

        # Fill spaces
        outer_space = self.__class__(outer_axes)
        for coord, entry in self.items():
            # Lookup subspace coord
            outer_coord = outer_space.coord_from_attrs(coord)
            try:
                # Get subspace
                inner_space = outer_space[outer_coord]
            except KeyError:
                # Create subspace, embed
                inner_space = self.__class__(inner_axes)
                outer_space[outer_coord] = inner_space
            # Add entry to subspace, similar to .fill()
            inner_space[inner_space.coord_from_attrs(coord)] = entry

        return outer_space

    def intersect_axes(self, attrs):
        """Rm those `a` in `attrs` that are not in `self.axes`.

        This allows errors in the axes allotment, for ease-of-use.
        """
        absent = struct_tools.complement(attrs, self.axes)
        if absent:
            print(color_text("Warning:", colorama.Fore.RED),
                  "The requested attributes",
                  color_text(str(absent), colorama.Fore.RED),
                  ("were not found among the"
                   " xpSpace axes (attrs. used as coordinates"
                   " for the set of experiments)."
                   " This may be no problem if the attr. is redundant"
                   " for the coord-sys."
                   " However, if it is caused by confusion or mis-spelling,"
                   " then it is likely to cause mis-interpretation"
                   " of the shown results."))
            attrs = struct_tools.complement(attrs, absent)
        return attrs

    def append_axis(self, axis):
        self.__init__(self.axes+(axis,))
        for coord in list(self):
            entry = self.pop(coord)
            self[coord + (None,)] = entry

    def label_xSection(self, label, *NoneAttrs, **sub_coord):
        """Insert duplicate entries for the given cross-section.

        Works by adding the attr. `xSection` to the axes of `SparseSpace`,
        and setting it to `label` for entries matching `sub_coord`,
        reflecting the "constance/constraint/fixation" this represents.
        This distinguishes the entries in this fixed-affine subspace,
        preventing them from being gobbled up by the operations of `nest`.

        If you wish, you can specify the `NoneAttrs`,
        which are consequently set to None for the duplicated entries,
        preventing them from being shown in plot labels and tuning panels.
        """
        if "xSect" not in self.axes:
            self.append_axis('xSect')

        for coord in self.coords_matching(**self.intersect_axes(sub_coord)):
            entry = copy.deepcopy(self[coord])
            coord = coord._replace(xSect=label)
            coord = coord._replace(**{a: None for a in NoneAttrs})
            self[coord] = entry


AXES_ROLES = dict(outer=None, inner=None, mean=None, optim=None)


class xpSpace(SparseSpace):
    """Functionality to facilitate working with `xps` and their results.

    `xpSpace.from_list` initializes a `SparseSpace` from a list
    of objects, typically experiments referred to as `xp`s, by

    - computing the relevant `axes` from the attributes, and
    - filling the dict by `xp`s.
    - computing and writing the attribute `ticks`.

    Using `xpSpace.from_list(xps)` creates a SparseSpace holding `xp`s.
    However, the nested `xpSpace`s output by `xpSpace.table_tree` will hold
    objects of type `UncertainQtty`,
    coz `xpSpace.table_tree` calls `mean` calls `get_stat(statkey)`.

    The main use of `xpSpace` is through `xpSpace.print` & `xpSpace.plot`,
    both of which call `xpSpace.table_tree` to nest the axes of the `SparseSpace`.
    """

    _ordering = dict(
        rot       = 'as_found',
        da_method = 'as_found',
    )

    @classmethod
    def from_list(cls, xps, ordering=None):
        """Init xpSpace from xpList."""

        def make_ticks(axes):
            """Unique & sort, for each axis (individually) in axes."""
            for ax_name, arr in axes.items():
                ticks = set(arr)  # unique (jumbles order)
                order = {**cls._ordering, **(ordering or {})}
                order = order.get(ax_name, 'default').lower()

                # Sort key
                if callable(order):
                    key = order
                elif 'as_found' in order:
                    key = arr.index
                else:
                    def key(x):
                        return x

                # Place None's at the end
                def key_safe(x):
                    return (x is None), key(x)

                # Sort
                ticks = sorted(ticks, key=key_safe)
                # Reverse
                if isinstance(order, str) and "rev" in order:
                    ticks = ticks[::-1]
                # Assign
                axes[ax_name] = ticks

        # Define and fill SparseSpace
        xp_list = xpList(xps)
        axes = xp_list.prep_table(nomerge=['xSect'])[0]
        self = cls(axes)
        self.fill(xps)

        make_ticks(axes)
        # Note: this attr (ticks) will not be propagated through nest().
        # That is fine. Otherwise we should have to prune the ticks
        # (if they are to be useful), which we don't want to do.
        self.ticks = axes

        return self

    def fill(self, xps):
        """Mass insertion."""
        self.update([(self.coord_from_attrs(xp), xp) for xp in xps])

    def squeeze(self):
        """Eliminate unnecessary axes/dimensions."""
        squeezed = xpSpace(xpList(self).prep_table()[0])
        squeezed.fill(self)
        return squeezed

    def get_stat(self, statkey="rmse.a"):
        """Make `xpSpace` with identical `keys`, but values `xp.avrgs.statkey`."""
        # Init a new xpDict to hold stat
        avrgs = self.__class__(self.axes)

        found_anything = False
        for coord, xp in self.items():
            val = getattr(xp.avrgs, statkey, None)
            avrgs[coord] = val
            found_anything = found_anything or (val is not None)

        if not found_anything:
            raise AttributeError(
                f"The stat.'{statkey}' was not found among any of the xp's.")

        return avrgs

    def mean(self, axes=None):
        # Note: The case `axes=()` should work w/o special treatment.
        if axes is None:
            return self

        nested = self.nest(axes)
        for coord, space in nested.items():

            def getval(uq): return uq.val if isinstance(uq, UncertainQtty) else uq
            vals = [getval(uq) for uq in space.values()]

            # Don't use nanmean! It would give false impressions.
            mu = np.mean(vals)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                # Don't print warnings caused by N=1.
                # It already correctly yield nan's.
                var = np.var(vals, ddof=1)

            N = len(vals)
            uq = UncertainQtty(mu, np.sqrt(var/N))
            uq.nTotal   = N
            uq.nFail    = N - np.isfinite(vals).sum()
            uq.nSuccess = N - uq.nFail

            nested[coord] = uq
        return nested

    def tune(self, axes=None, costfun=None):
        """Get (compile/tabulate) a stat. optimised wrt. tuning params."""
        # Define cost-function
        costfun = (costfun or 'increasing').lower()
        if 'increas' in costfun:
            costfun = (lambda x: +x)
        elif 'decreas' in costfun:
            costfun = (lambda x: -x)
        else:
            assert callable(costfun)  # custom

        # Note: The case `axes=()` should work w/o special treatment.
        if axes is None:
            return self

        nested = self.nest(axes)
        for coord, space in nested.items():

            # Find optimal value and coord within space
            MIN = np.inf
            for inner_coord, uq in space.items():
                cost = costfun(uq.val)
                if cost <= MIN:
                    MIN                = cost
                    uq_opt             = uq
                    uq_opt.tuned_coord = inner_coord

            nested[coord] = uq_opt

        return nested

    def validate_axes(self, axes):
        """Validate axes.

        Note: This does not convert None to (),
              allowing None to remain special.
              Use `axis or ()` wherever tuples are required.
        """
        roles = {}  # "inv"
        for role in set(axes) | set(AXES_ROLES):
            assert role in AXES_ROLES, f"Invalid role {role!r}"
            aa = axes.get(role, AXES_ROLES[role])

            if aa is None:
                pass  # Purposely special
            else:
                # Ensure iterable
                if isinstance(aa, str) or not hasattr(aa, "__iter__"):
                    aa = (aa,)

                aa = self.intersect_axes(aa)

                for axis in aa:

                    # Ensure unique
                    if axis in roles:
                        raise TypeError(
                            f"An axis (here {axis!r}) cannot be assigned to 2"
                            f" roles (here {role!r} and {roles[axis]!r}).")
                    else:
                        roles[axis] = role
            axes[role] = aa
        return axes

    def table_tree(self, statkey, axes, *, costfun=None):
        """Hierarchical nest(): xp_dict>outer>inner>mean>optim.

        as specified by `axes`. Returns this new xpSpace.

        - print_1d / plot_1d (respectively) separate
          tables / panel(row)s for `axes['outer']`, and
          columns/ x-axis      for `axes['inner']`.

        - The `axes['mean']` and `axes['optim']` get eliminated
          by the mean()/tune() operations.

        Note: cannot support multiple statkeys
              because it's not (obviously) meaningful
              when optimizing over tuning_axes.
        """
        axes = self.validate_axes(axes)

        def mean_tune(xp_dict):
            """Take mean, then tune.

            Note: the SparseDict implementation should be sufficiently
            "uncluttered" that mean_tune() (or a few of its code lines)
            could be called anywhere above/between/below
            the `nest()`ing of `outer` or `inner`.
            These possibile call locations are commented in the code.
            """
            uq_dict = xp_dict.get_stat(statkey)
            uq_dict = uq_dict.mean(axes['mean'])
            uq_dict = uq_dict.tune(axes['optim'], costfun)
            return uq_dict

        self = mean_tune(self)
        # Prefer calling mean_tune() [also see its docstring]
        # before doing outer/inner nesting. This is because then the axes of
        # a row (xpSpace) should not include mean&optim, and thus:
        #  - Column header/coords may be had directly as row.keys(),
        #    without extraction by coord_from_attrs() from (e.g.) row[0].
        #  - Don't need to propagate mean&optim axes down to the row level.
        #    which would require defining rows by the nesting:
        #    rows = table.nest(outer_axes=struct_tools.complement(table.axes,
        #        *(axes['inner'] or ()),
        #        *(axes['mean']  or ()),
        #        *(axes['optim'] or ()) ))
        #  - Each level of the output from table_tree
        #    is a smaller (and more manageable) dict.

        tables = self.nest(outer_axes=axes['outer'])
        for table_coord, table in tables.items():
            # table = mean_tune(table)

            # Should not be used (nesting as rows is more natural,
            # and is required for getting distinct/row_keys).
            # cols = table.nest(outer_axes=axes['inner'])

            rows = table.nest(inner_axes=axes['inner'] or ())

            # Overwrite table by its nesting as rows
            tables[table_coord] = rows

            # for row_coord, row in rows.items():
            # rows[row_coord] = mean_tune(row)

        return axes, tables

    def tickz(self, axis_name):
        """Axis ticks without None"""
        return [x for x in self.ticks[axis_name] if x is not None]

    def print(self, statkey="rmse.a", axes=AXES_ROLES,  # noqa
              subcols=True, decimals=None, costfun=None,
              squeeze_labels=True, colorize=True):
        """Print tables of results.

        Parameters
        ----------
        statkey: str
            The statistic to extract from the `xp.avrgs` for each `xp`.
        axes: dict
            Allots (maps) the axes/dimensions of `xpSpace` to different
            roles in the printing of the table.

            - Herein, the "role" `outer` should list the axes/attributes
              used to define the splitting of the results into *separate tables*:
              one table for each distinct combination of attributes.
            - Similarly , the role `inner` determines which attributes
              split a table into its columns.
            - `mean` lists the attributes over which the mean is taken
              (for that row & column)
            - `optim` lists the attributes used over which the optimum
               is searched for (after taking the mean).

            Example:

                dict(outer='da_method', inner='N', mean='seed',
                     optim=('infl','loc_rad'))

            Equivalently, use `mean=("seed",)`.
            It is acceptible to not specify anything, e.g. `mean=()` or `mean=None`.
        subcols: bool
            If `True`, then subcolumns are added to indicate

            - `1σ`: the confidence interval. If `mean=None` is used, this simply reports
              the value `.prec` of the `statkey`, providing this is an `UncertainQtty`.
              Otherwise, it is computed as `sqrt(var(xps)/N)`,
              where `xps` is the set of statistic gathered over the `mean` axis.
            - `*(optim)`: the optimal point (among all `optim` attributes),
              as defined by `costfun`.
            - `☠`: the number of failures (non-finite values) at that point.
            - `✓`: the number of successes that go into the value
        decimals: int
            Number of decimals to print.
            If `None`, this is determined for each statistic by its uncertainty.
        costfun: str or function
            Use `'increasing'` (default) or `'decreasing'` to indicate that the optimum
            is defined as the lowest or highest value of the `statkey` found.
        squeeze_labels: bool
            Don't include redundant attributes in the line labels.
            Caution: `get_style` will not be able to access the eliminated attrs.
        colorize: bool
            Add color to tables for readability.
        """
        # Inform axes["mean"]
        if axes.get('mean', None):
            print(f"Averages (in time and) over {axes['mean']}.")
        else:
            print("Averages in time only"
                  " (=> the 1σ estimates may be unreliable).")

        axes, tables = self.table_tree(statkey, axes, costfun=costfun)

        def make_cols(rows, cc, subcols, h2):
            """Subcolumns: align, justify, join."""
            # Define subcol formats
            if subcols:
                templ = "{val} ±{prec}"
                templ += "" if axes['optim'] is None else " *{tuned_coord}"
                templ += "" if  axes['mean'] is None else " {nFail} {nSuccess}"  # noqa
                aligns = dict(prec="<", tuned_coord="<")
                labels = dict(val=statkey, prec="1σ",
                              tuned_coord=axes["optim"],
                              nFail="☠", nSuccess="✓")

            def align(column):
                col = unpack_uqs(column, decimals)
                if subcols:
                    for key in list(col):
                        if key in templ:
                            subcolmn = [labels.get(key, key)] + col[key]
                            col[key] = align_col(subcolmn, just=aligns.get(key, ">"))
                        else:
                            del col[key]
                    col = [templ.format(**row) for row in struct_tools.transps(col)]
                else:
                    col = align_col([statkey] + col["val"])
                return col

            def super_header(col_coord, idx, col):
                header, matter = col[0], col[1:]
                if idx:
                    cc = str(col_coord).strip("()")
                else:
                    cc = col_coord.repr2()
                cc = cc.replace(", ", ",")
                cc = cc.center(len(header), "_")  # +1 width for wide chars like ✔️
                return [cc + "\n" + header] + matter

            # Transpose
            columns = [list(x) for x in zip(*rows)]

            # Format column
            for j, (col_coord, column) in enumerate(zip(cc, columns)):
                col = align(column)
                if h2:
                    col = super_header(col_coord, j, col)
                columns[j] = col

            # Un-transpose
            rows = [list(x) for x in zip(*columns)]

            return rows

        for table_coord, table in tables.items():

            # Get table's column coords
            # It's supposed to be a set, but we use a dict to keep ordering.
            # cc = self.ticks[axes["inner"]]  # may be > needed
            # cc = table[0].keys()            # may be < needed
            cc = {c: None for row in table.values() for c in row}
            # Could also do cc = table.squeeze() but is it worth it?

            # Convert table (rows) into rows (lists) of equal length
            rows = [[row.get(c, None) for c in cc] for row in table.values()]

            h2 = "\n" if len(cc) > 1 else ""  # super-header?
            rows = make_cols(rows, cc, subcols, h2)

            if squeeze_labels:
                table = table.squeeze()

            # Prepend left-side (attr) table
            # Header
            rows[0] = [h2+k for k in table.axes] + [h2+'⑊'] + rows[0]
            # Matter
            for i, (key, row) in enumerate(zip(table, rows[1:])):
                rows[i+1] = [*key] + ['|'] + row

            # Print
            print("\n", end="")
            if axes['outer']:
                table_title = "Table for " + table_coord.repr2()
                if colorize:
                    clrs = colorama.Back.YELLOW, colorama.Fore.BLACK
                    table_title = color_text(table_title, *clrs)
                print(table_title)
            headers, *rows = rows
            t = tabulate(rows, headers).replace('␣', ' ')
            if colorize:
                t = stripe(t, slice(2, None))
            print(t)

    def plot(self, statkey="rmse.a", axes=AXES_ROLES, get_style=default_styles,
             fignum=None, figsize=None, panels=None,
             title2=None, costfun=None, unique_labels=True,
             squeeze_labels=True):
        """Plot (tables of) results.

        Analagously to `xpSpace.print`,
        the averages are grouped by `axis["inner"]`,
        which here plays the role of the x-axis.

        The averages can also be grouped by `axis["outer"]`,
        producing a figure with multiple (columns of) panels.

        The optimal points/parameters/attributes are plotted in smaller panels
        below the main plot. This can be turned off by providing the figure
        axes through the `panels` argument.

        The parameters `statkey`, `axes`, `costfun`, `sqeeze_labels`
        are documented in `xpSpace.print`.

        Parameters
        ----------
        get_style: function
            A function that takes an object, and returns a dict of line styles,
            usually as a function of the object's attributes.
        title2: str
            Figure title (in addition to the defaults).
        unique_labels: bool
            Only show a given label once.
        """
        def plot1(panelcol, row, style):
            """Plot a given line (row) in the main panel and the optim panels.

            Involves: Sort, insert None's, handle constant lines.
            """
            # Make a full row (yy) of vals, whether is_constant or not.
            # row.is_constant = (len(row)==1 and next(iter(row))==row.Coord(None))
            row.is_constant = all(x == row.Coord(None) for x in row)
            yy = [row[0] if row.is_constant else y for y in row.get_for(xticks)]

            # Plot main
            row.vals = [getattr(y, 'val', None) for y in yy]
            row.handles = {}
            row.handles["main_panel"] = panelcol[0].plot(xticks, row.vals, **style)[0]

            # Plot tuning params
            row.tuned_coords = {}  # Store ordered, "transposed" argmins
            argmins = [getattr(y, 'tuned_coord', None) for y in yy]
            for a, panel in zip(axes["optim"], panelcol[1:]):
                yy = [getattr(coord, a, None) for coord in argmins]
                row.tuned_coords[a] = yy

                # Plotting all None's sets axes units (like any plotting call)
                # which can cause trouble if the axes units were actually supposed
                # to be categorical (eg upd_a), but this is only revealed later.
                if not all(y == None for y in yy):
                    row.handles[a] = panel.plot(xticks, yy, **style)

        # Nest axes through table_tree()
        assert len(axes["inner"]) == 1, "You must chose the abscissa."
        axes, tables = self.table_tree(statkey, axes, costfun=costfun)
        xticks = self.tickz(axes["inner"][0])

        # Create figure panels
        if panels is None:
            nrows   = len(axes['optim'] or ()) + 1
            ncols   = len(tables)
            maxW    = 12.7  # my mac screen
            figsize = figsize or (min(5*ncols, maxW), 7)
            gs      = dict(
                height_ratios=[6]+[1]*(nrows-1),
                hspace=0.05, wspace=0.05,
                # eyeballed:
                left=0.15/(1+np.log(ncols)),
                right=0.97, bottom=0.06, top=0.9)
            # Create
            _, panels = place.freshfig(num=fignum, figsize=figsize,
                                       nrows=nrows, sharex=True,
                                       ncols=ncols, sharey='row',
                                       gridspec_kw=gs, squeeze=False)
        else:
            panels = np.atleast_2d(panels)

        # Fig. Title
        fig = panels[0, 0].figure
        fig_title = "Averages wrt. time"
        if axes["mean"] is not None:
            fig_title += " and " + ", ".join([repr(c) for c in axes['mean']])
        if title2 is not None:
            with nonchalance():
                title2 = title2.relative_to(rc.dirs["data"])
            fig_title += "\n" + str(title2)
        fig.suptitle(fig_title)

        # Loop outer
        label_register = set()  # mv inside loop to get legend on each panel
        for table_panels, (table_coord, table) in zip(panels.T, tables.items()):
            table.panels = table_panels
            title = "" if axes["outer"] is None else table_coord.repr2()

            if squeeze_labels:
                distinct = xpList(table.keys()).squeeze()[0]
            else:
                distinct = table.axes

            # Plot
            for coord, row in table.items():

                coord = NoneDict(struct_tools.intersect(coord._asdict(), distinct))
                style = get_style(coord)

                # Rm duplicate labels
                if unique_labels:
                    if style.get("label", None) in label_register:
                        del style["label"]
                    else:
                        label_register.add(style["label"])

                plot1(table.panels, row, style)

            # Beautify
            panel0 = table.panels[0]
            # panel0.set_title(title)
            panel0.text(.5, 1, title, fontsize=12, ha="center", va="bottom",
                        transform=panel0.transAxes, bbox=dict(
                            facecolor='lightyellow', edgecolor='k',
                            alpha=0.99, boxstyle="round,pad=0.25",
                            # NB: padding makes label spill into axes
                        ))
            if panel0.is_first_col():
                panel0.set_ylabel(statkey)
            with set_tmp(mpl_logger, 'level', 99):  # silence "no label" msg
                panel0.legend()
            table.panels[-1].set_xlabel(axes["inner"][0])
            # Tuning panels:
            for a, panel in zip(axes["optim"] or (), table.panels[1:]):
                if panel.is_first_col():
                    panel.set_ylabel(f"Optim.\n{a}")

        tables.fig = fig
        tables.xp_dict = self
        tables.axes_roles = axes
        return tables


def default_fig_adjustments(tables, xticks_from_data=False):
    """Beautify. These settings do not generalize well."""
    # Get axs as 2d-array
    axs = np.array([table.panels for table in tables.values()]).T

    # Main panels (top row) only:
    sensible_f = ticker.FormatStrFormatter('%g')
    for ax in axs[0, :]:  # noqa
        for direction in ['y', 'x']:
            if axs.shape[1] < 6:
                eval(f"ax.set_{direction}scale('log')")
                eval(f"ax.{direction}axis").set_minor_formatter(sensible_f)
            eval(f"ax.{direction}axis").set_major_formatter(sensible_f)

    # Set xticks
    if xticks_from_data:
        ax = tables[1].panels[0]
        # Log-scale overrules any custom ticks. Restore control
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xticks(tables.xp_dict.tickz(tables.axes_roles["inner"][0]))

    # Tuning panels only
    table = tables[0]
    for a, panel in zip(tables.axes_roles["optim"] or (), table.panels[1:]):
        yy = tables.xp_dict.tickz(a)
        axis_scale_by_array(panel, yy, "y")
        # set_ymargin doesn't work for wonky scales. Do so manually:
        alpha = len(yy)/10
        y0, y1, y2, y3 = yy[0], yy[1], yy[-2], yy[-1]
        panel.set_ylim(y0-alpha*(y1-y0), y3+alpha*(y3-y2))

    # All panels
    for ax in axs.ravel():
        for direction in ['y', 'x']:
            if axs.shape[1] < 6:
                ax.grid(True, which="minor", axis=direction)

    # Not strictly compatible with gridspec height_ratios,
    # (throws warning), but still works ok.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        axs[0, 0].figure.tight_layout()
