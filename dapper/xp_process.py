"""Tools (notably `xpSpace`) for processing and presenting experiment data."""

import collections
import copy
import logging
import os
import shutil
import warnings
from pathlib import Path

import colorama
import dill
import matplotlib as mpl
import numpy as np
import struct_tools
from matplotlib import cm, ticker
from patlib.std import set_tmp
from tabulate import tabulate
from tqdm import tqdm

import dapper.tools.remote.uplink as uplink
from dapper.stats import align_col, unpack_uqs
from dapper.tools.colors import color_text
from dapper.tools.series import UncertainQtty
from dapper.tools.viz import axis_scale_by_array, freshfig
from dapper.xp_launch import collapse_str, xpList

mpl_logger = logging.getLogger('matplotlib')

NO_KEY = ("da_method", "Const", "upd_a")


def make_label(coord, no_key=NO_KEY, exclude=()):
    dct = {a: v for a, v in coord._asdict().items() if v != None}
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
    style = struct_tools.DotDict(ms=8)
    style.label = make_label(coord)

    try:
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

    except AttributeError:
        pass

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
    """Discretize `cmap` so that it partitions `[0,1]` into `N` segments.

    I.e. `cmap(k/N) == cmap(k/N + eps)`.

    Also provide the ScalarMappable `sm`
    that maps range(N) to the segment centers,
    as will be reflected by `cb = fig.colorbar(sm)`.
    You can then re-label the ticks using
    `cb.set_ticks(np.arange(N)); cb.set_ticklabels(["A","B","C",...])`."""
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

    Example: rename attr n_iter to nIter:
    >>> proj_name = "Stein"
    >>> dd = dpr.rc.dirs.data / proj_name
    >>> save_as = dd / "run_2020-09-22__19:36:13"
    >>>
    >>> for save_as in os.listdir(dd):
    >>>     save_as = dd / save_as
    >>>
    >>>     xps = load_xps(save_as)
    >>>     HMM = load_HMM(save_as)
    >>>
    >>>     for xp in xps:
    >>>         if hasattr(xp,"n_iter"):
    >>>             xp.nIter = xp.n_iter
    >>>             del xp.n_iter
    >>>
    >>>     overwrite_xps(xps, save_as)
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
    This is probably also the reason that the loading time is sometimes reduced."""
    overwrite_xps(load_xps(save_as), save_as, nDir)


class SparseSpace(dict):
    """Subclass of `dict` that enforces key conformity to a `namedtuple`.

    Like a normal `dict`, it can hold any type of objects.
    But, since keys must conform, this effectively defines a coordinate system,
    i.e. vector **space**.

    The coordinate system is specified by its "axes",
    which is used to produce `self.Coord` (a `namedtuple` class).

    In normal use, this space is highly sparse,
    coz there are many coordinates with no matching experiment,
    eg. `coord(da_method=Climatology, rot=True, ...)`.

    Indeed, operations across (potentially multiple) axes,
    such as optimization or averaging, should be carried out by iterating
    -- not over the axes -- but over the the list of items.

    The most important method is `nest`,
    which is used (by `xpSpace.table_tree`) to separate tables/columns,
    and also to carry out the mean/optim operations.

    In addition, `__getitem__` is very flexible, allowing accessing by:

    - The actual key, a `self.Coord` object. Returns single item.
    - A `dict` to match against (part of) the coordinates. Returns subspace.
    - An `int`. Returns `list(self)[key]`.
    - A list of any of the above. Returns list.

    This flexibility can cause bugs, but it's probably still worth it.
    Also see `__call__`, `get_for`, and `coords`,
    for further convenience.

    Inspired by

    - https://stackoverflow.com/a/7728830
    - https://stackoverflow.com/q/3387691
    """

    @property
    def axes(self):
        return self.Coord._fields

    def __init__(self, axes, *args, **kwargs):
        """Usually initialized through `xpSpace`.

        Parameters
        ----------
        axes: list
            The attributes defining the coordinate system.

        args: entries
            Nothing, or a list of `xp`s.
        """
        # Define coordinate system
        self.Coord = collections.namedtuple('Coord', axes)
        # Write dict
        self.update(*args, **kwargs)
        # Add repr/str
        self.Coord.__repr__ = lambda c: ",".join(
            f"{k}={v!r}" for k, v in zip(c._fields, c))
        self.Coord.__str__  = lambda c: ",".join(str(v) for v in c)

    def update(self, *args, **kwargs):
        """Update using custom `__setitem__`."""
        # See https://stackoverflow.com/a/2588648
        # and https://stackoverflow.com/a/2390997
        for k, v in dict(*args, **kwargs).items():
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
            coord = outer.Coord(*key.values())      # create coord
            inner = outer[coord]                    # chose subspace
            return inner

        # Single item (by Coord object, coz an integer (eg)
        # gets interpreted (above) as a list index)
        else:
            # NB: Dont't use isinstance(key, self.Coord)
            # coz it fails when the namedtuple (Coord) has been
            # instantiated in different places (but with equal params).
            # Also see bugs.python.org/issue7796
            return super().__getitem__(key)

    def __getkey__(self, entry):
        """Inverse of `dict.__getitem__`, but also works on coords.

        Note: This dunder method is not a "builtin" naming convention."""
        coord = (getattr(entry, a, None) for a in self.axes)
        return self.Coord(*coord)

    def __call__(self, **kwargs):
        """Convenience, that enables, eg.:
        >>> xp_dict(da_method="EnKF", infl=1, seed=3)
        """
        return self.__getitem__(kwargs)

    def get_for(self, ticks, default=None):
        """Almost `[self.get(Coord(x)) for x in ticks]`.

        NB: using the "naive" thing: `[self[x] for x in ticks]`
        would probably be a BUG coz x gets interpreted as indices
        for the internal list."""
        singleton = not hasattr(ticks[0], "__iter__")
        def coord(xyz): return self.Coord(xyz if singleton else xyz)
        return [self.get(coord(x), default) for x in ticks]

    def coords(self, **kwargs):
        """Get all `coord`s matching kwargs.

        Unlike `__getitem__(kwargs)`,
        - A list is returned, not a subspace.
        - This list constains keys (coords), not values.
        - The coords refer to the original space, not the subspace.

        The last point is especially useful for
        `SparseSpace.label_xSection`.
        """

        def embed(coord): return {**kwargs, **coord._asdict()}
        return [self.Coord(**embed(x)) for x in self[kwargs]]

        # Old implementation.
        # - I prefer the new version for its re-use of __getitem__'s
        #   nesting, evidencing their mutual relationship)
        # - Note that unlike xpList.inds(): missingval shenanigans
        #   are here unnecessary coz each coordinate is complete.
        # match  = lambda x: all(getattr(x,k)==kwargs[k] for k in kwargs)
        # return [x for x in self if match(x)]

    def __repr__(self):
        # Note: print(xpList(self)) produces more human-readable key listing,
        # but we don't want to implement it here, coz it requires split_attrs(),
        # which we don't really want to call again.
        L = 2
        keys = [str(k) for k in self]
        if 2*L < len(keys):
            keys = keys[:L] + ["..."] + keys[-L:]
        keys = "[\n  " + ",\n  ".join(keys) + "\n]"
        txt = f"<{self.__class__.__name__}> with {len(self)} keys: {keys}"
        # txt += " befitting the coord. sys. with axes "
        txt += "\nplaced in a coord-sys with axes "
        try:
            txt += "(and ticks):" + str(struct_tools.AlignedDict(self.ticks))
        except AttributeError:
            txt += ":\n" + str(self.axes)
        return txt

    def nest(self, inner_axes=None, outer_axes=None):
        """Return a new xpSpace with axes `outer_axes`,

        obtained by projecting along the `inner_axes`.
        The entries of this `xpSpace` are themselves `xpSpace`s,
        with axes `inner_axes`,
        each one regrouping the entries with the same (projected) coordinate.

        Note: is also called by `__getitem__(key)` if `key` is dict."""

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
            outer_coord = outer_space.__getkey__(coord)
            try:
                inner_space = outer_space[outer_coord]
            except KeyError:
                inner_space = self.__class__(inner_axes)
                outer_space[outer_coord] = inner_space
            inner_space[inner_space.__getkey__(coord)] = entry

        return outer_space

    def add_axis(self, axis):
        self.__init__(self.axes+(axis,))
        for coord in list(self):
            entry = self.pop(coord)
            self[coord + (None,)] = entry

    def intersect_axes(self, attrs):
        """Rm those a in attrs that are not in self.axes.

        This allows errors in the axes allotment, for ease-of-use."""
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

    def label_xSection(self, label, *NoneAttrs, **sub_coord):
        """Insert duplicate entries for the cross section

        whose `coord`s match `sub_coord`,
        adding the attr `Const=label` to their `coord`,
        reflecting the "constance/constraint/fixation" this represents.

        This distinguishes the entries in this fixed-affine subspace,
        preventing them from being gobbled up in `nest`.

        If you wish, you can specify the `NoneAttrs`,
        which are consequently set to None for the duplicated entries,
        preventing them from getting plotted in tuning panels.
        """

        if "Const" not in self.axes:
            self.add_axis('Const')

        for coord in self.coords(**self.intersect_axes(sub_coord)):
            entry = copy.deepcopy(self[coord])
            coord = coord._replace(Const=label)
            coord = coord._replace(**{a: None for a in NoneAttrs})
            self[coord] = entry


AXES_ROLES = dict(outer=None, inner=None, mean=None, optim=None)


class xpSpace(SparseSpace):
    """Functionality to facilitate working with `xps` and their results.

    `xpSpace.from_list` initializes a `SparseSpace` from a list
    of objects, typically experiments referred to as `xp`s, by
    (1) computing the relevant axes from the attributes, and
    (2) filling the dict by `xp`s.

    Using `xpSpace.from_list(xps)` creates a SparseSpace holding `xp`s.
    However, the nested `xpSpace`s output by `xpSpace.table_tree` will hold
    objects of type `UncertainQtty`,
    coz `xpSpace.table_tree` calls `mean` calls `field(statkey)`.

    The main use of `xpSpace` is through `xpSpace.print` & `xpSpace.plot`,
    both of which call `xpSpace.table_tree` to nest the axes of the `SparseSpace`.
    """

    @classmethod
    def from_list(cls, xps):
        """Init xpSpace from xpList."""

        def make_ticks(axes, ordering=dict(
                    N         = 'default',
                    seed      = 'default',
                    infl      = 'default',
                    loc_rad   = 'default',
                    rot       = 'as_found',
                    da_method = 'as_found',
                    )):
            """Unique & sort, for each axis (individually) in axes."""

            for ax_name, arr in axes.items():
                ticks = set(arr)  # unique (jumbles order)

                # Sort
                order = ordering.get(ax_name, 'default').lower()
                if hasattr(order, '__call__'):  # eg. mylist.index
                    ticks = sorted(ticks, key=order)
                elif 'as_found' in order:
                    ticks = sorted(ticks, key=arr.index)
                else:  # default sorting, with None placed at the end
                    ticks = sorted(ticks, key= lambda x: (x is None, x))
                if any(x in order for x in ['rev', 'inv']):
                    ticks = ticks[::-1]
                axes[ax_name] = ticks

        # Define axes
        xp_list = xpList(xps)
        axes = xp_list.split_attrs(nomerge=['Const'])[0]
        make_ticks(axes)
        self = cls(axes.keys())

        # Note: this attr (ticks) will not be propagated through nest().
        # That is fine. Otherwise we should have to prune the ticks
        # (if they are to be useful), which we don't want to do.
        self.ticks = axes

        # Fill
        self.update({self.__getkey__(xp): xp for xp in xps})

        return self

    def field(self, statkey="rmse.a"):
        """Extract `statkey` for each item in `self`."""

        # Init a new xpDict to hold field
        avrgs = self.__class__(self.axes)

        found_anything = False
        for coord, xp in self.items():
            val = getattr(xp.avrgs, statkey, None)
            avrgs[coord] = val
            found_anything = found_anything or (val is not None)

        if not found_anything:
            raise AttributeError(
                f"The stat. field '{statkey}' was not found"
                " among any of the xp's.")

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
        """Get (compile/tabulate) a stat field optimised wrt. tuning params."""

        # Define cost-function
        costfun = (costfun or 'increasing').lower()
        if 'increas' in costfun:
            costfun = (lambda x: +x)
        elif 'decreas' in costfun:
            costfun = (lambda x: -x)
        else:
            assert hasattr(costfun, '__call__')  # custom

        # Note: The case `axes=()` should work w/o special treatment.
        if axes is None:
            return self

        nested = self.nest(axes)
        for coord, space in nested.items():

            # Find optimal value and coord within space
            MIN = np.inf
            for i, (inner_coord, uq) in enumerate(space.items()):
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

    def table_tree(self, statkey, axes):
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
            These possibile call locations are commented in the code."""
            uq_dict = xp_dict.field(statkey)
            uq_dict = uq_dict.mean(axes['mean'])
            uq_dict = uq_dict.tune(axes['optim'])
            return uq_dict

        self = mean_tune(self)
        # Prefer calling mean_tune() [also see its docstring]
        # before doing outer/inner nesting. This is because then the axes of
        # a row (xpSpace) should not include mean&optim, and thus:
        #  - Column header/coords may be had directly as row.keys(),
        #    without extraction by __getkey__() from (e.g.) row[0].
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

    def print(self, statkey="rmse.a", axes=AXES_ROLES,
              subcols=True, decimals=None):
        """Print tables of results.

        Parameters
        ----------
        statkey: str
            The statistical field from the experiments to report.
        subcols: bool
            If `True`, then subcolumns are added to indicate the
            1σ confidence interval, and potentially some other stuff.
        axes: dict
            Allots (maps) each role to a set of axis of the `xpSpace`.

                >>> dict(
                >>>    outer='da_method', inner='N', mean='seed',
                >>>    optim=('infl','loc_rad'))

            - Herein, the "role" `outer` should list the axes/attributes
            used to define the splitting of the results into *separate tables*:
            one table for each distinct (combination) of attributes.
            - Similarly , the role `inner` determines which attributes
            split a table into its columns.
            - `mean` lists the attributes used over which the mean is taken.
            - `optim` lists the attributes used over which the optimum result
               is searched for.

            Example: If `mean` is assigned to:

            - `("seed",)`: Experiments are averaged accross seeds,
                           and the 1σ (sub)col is computed as sqrt(var(xps)/N),
                           where xps is a set of experiments.

            - `()`       : Experiments are averaged across nothing
                           (i.e. this is an edge case).

            - `None`     : Experiments are not averaged
                           (i.e. the values are the same as above),
                           and the 1σ (sub)col is computed from
                           the time series of that single experiment.
        decimals: int
            Number of decimals to print.
            If `None`, this is determined for each statistic by its uncertainty.
        """

        def align_subcols(rows, cc, subcols, h2):
            """Subcolumns: align, justify, join."""

            # Define subcol formats
            subc = dict()
            subc['keys']     = ["val", "conf"]
            subc['headers']  = [statkey, '1σ']
            subc['frmts']    = [None, None]
            subc['spaces']   = [' ±', ]  # last one gets appended below.
            subc['aligns']   = ['>', '<']  # 4 header -- matter gets decimal-aligned.
            if axes['optim'] is not None:
                subc['keys']    += ["tuned_coord"]
                subc['headers'] += [axes['optim']]
                subc['frmts']   += [lambda x: tuple(a for a in x)]
                subc['spaces']  += [' *']
                subc['aligns']  += ['<']
            elif axes['mean'] is not None:
                subc['keys']    += ["nFail", "nSuccess"]
                subc['headers'] += ['☠', '✓']  # use width-1 symbols!
                subc['frmts']   += [None, None]
                subc['spaces']  += [' ', ' ']
                subc['aligns']  += ['>', '>']
            subc['spaces'].append('')  # no space after last subcol
            template = '{}' + '{}'.join(subc['spaces'])

            # Transpose
            columns = [list(x) for x in zip(*rows)]

            # Iterate over columns.
            for j, (col_coord, column) in enumerate(zip(cc, columns)):

                # Tabulate columns
                if subcols:
                    column = unpack_uqs(column, decimals, subc["keys"])
                    # Tabulate subcolumns
                    subheaders = []
                    for key, header, frmt, _, align in zip(*subc.values()):
                        column[key] = align_col(
                            column[key], header, frmt=frmt)[1:]
                        L = len(column[-1][key])
                        if align == '<':
                            subheaders += [str(header).ljust(L)]
                        else:
                            subheaders += [str(header).rjust(L)]
                    # Join subcolumns:
                    matter = [
                        template.format(*[row[k] for k in subc['keys']])
                        for row in column
                    ]
                    header = template.format(*subheaders)
                else:
                    column = unpack_uqs(column, decimals)["val"]
                    column = align_col(column, statkey)
                    header, matter = column[0], column[1:]

                if h2:  # Do super_header
                    if j:
                        super_header = str(col_coord)
                    else:
                        super_header = repr(col_coord)
                    width = len(header)  # += 1 if using unicode chars like ✔️
                    super_header = super_header.center(width, "_")
                    header = super_header + "\n" + header

                columns[j] = [header]+matter
            # Un-transpose
            rows = [list(x) for x in zip(*columns)]

            return rows

        # Inform axes["mean"]
        if axes.get('mean', None):
            print(f"Averages (in time and) over {axes['mean']}.")
        else:
            print("Averages in time only"
                  " (=> the 1σ estimates may be unreliable).")

        axes, tables = self.table_tree(statkey, axes)
        for table_coord, table in tables.items():

            # Get this table's column coords (cc). Use dict for sorted&unique.
            # cc = xp_dict.ticks[axes["inner"]] # May be larger than needed.
            # cc = table[0].keys()              # May be too small a set.
            cc = {c: None for row in table.values() for c in row}

            # Convert table (rows) into rows (lists) of equal length
            rows = [[row.get(c, None) for c in cc] for row in table.values()]

            if False:  # ****************** Simple (for debugging) table
                for i, (row_coord, row) in enumerate(zip(table, rows)):
                    row_key = ", ".join(str(v) for v in row_coord)
                    rows[i] = [row_key]         + row
                rows.insert(0, [f"{table.axes}"] + [repr(c) for c in cc])
            else:  # ********************** Elegant table.
                h2 = "\n" if len(cc) > 1 else ""  # do column-super-header
                rows = align_subcols(rows, cc, subcols, h2)

                # Make and prepend left-side table
                # - It's prettier if row_keys don't have unnecessary cols.
                #   For example, the table of Climatology should not have an
                #   entire column repeatedly displaying "infl=None".
                #   => split_attrs().
                # - Why didn't we do this for the column attrs?
                #   Coz there we have no ambition to split the attrs,
                #   which would also require excessive processing:
                #   nesting the table as cols, and then split_attrs() on cols.
                row_keys = xpList(table.keys()).split_attrs()[0]
                if len(row_keys):
                    # Header
                    rows[0] = [h2+k for k in row_keys] + [h2+'⑊'] + rows[0]
                    # Matter
                    for i, (row, key) in enumerate(zip(
                            rows[1:], struct_tools.transps(row_keys))):
                        rows[i+1] = [*key.values()] + ['|'] + row

            # Print
            print("\n", end="")
            if axes['outer']:
                table_title = "Table for " + repr(table_coord)
                print(color_text(table_title, colorama.Back.YELLOW))
            headers, *rows = rows
            print(tabulate(rows, headers).replace('␣', ' '))

    def plot(self, statkey="rmse.a", axes=AXES_ROLES, get_style=default_styles,
             fignum=None, figsize=None, panels=None,
             title2=None, costfun=None, unique_labels=True):
        """Plot the avrgs of `statkey` as a function of `axis["inner"]`.

        Optionally, the experiments can be grouped by `axis["outer"]`,
        producing a figure with columns of panels.
        Firs of all, though, mean and optimum computations are done for
        `axis["mean"]` and `axis["optim"]`, where the optimization can
        be controlled through `costfun` (see `xpSpace.tune`)

        This is entirely analogous to the roles of `axis` in `xpSpace.print`.

        The optimal parameters are plotted in smaller panels below the main plot.
        This can be prevented by providing the figure axes through the `panels` arg.
        """

        def plot1(panelcol, row, style):
            """Plot a given line (row) in the main panel and the optim panels.

            Involves: Sort, insert None's, handle constant lines."""

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
        axes, tables = self.table_tree(statkey, axes)
        xticks = self.tickz(axes["inner"][0])

        # Figure panels
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
            _, panels = freshfig(num=fignum, figsize=figsize,
                                 nrows=nrows, sharex=True,
                                 ncols=ncols, sharey='row',
                                 gridspec_kw=gs)
            panels = np.ravel(panels).reshape((-1, ncols))
        else:
            panels = np.atleast_2d(panels)

        # Title
        fig = panels[0, 0].figure
        fig_title = "Average wrt. time"
        if axes["mean"] is not None:
            fig_title += f" and {axes['mean']}"
        if title2 is not None:
            fig_title += "\n" + str(title2)
        fig.suptitle(fig_title)

        # Loop outer
        label_register = set()  # mv inside loop to get legend on each panel
        for table_panels, (table_coord, table) in zip(panels.T, tables.items()):
            table.panels = table_panels
            title = '' if axes["outer"] is None else repr(table_coord)

            # Plot
            for coord, row in table.items():
                style = get_style(coord)

                # Rm duplicate labels (contrary to coords, labels can
                # be "tampered" with, and so can be duplicate)
                if unique_labels:
                    if style.get("label", None) in label_register:
                        del style["label"]
                    else:
                        label_register.add(style["label"])

                plot1(table.panels, row, style)

            # Beautify
            panel0 = table.panels[0]
            panel0.set_title(title)
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


def default_fig_adjustments(tables):
    """Beautify. These settings do not generalize well."""
    # Get axs as 2d-array
    axs = np.array([table.panels for table in tables.values()]).T

    # Main panels (top row) only:
    sensible_f = ticker.FormatStrFormatter('%g')
    for ax in axs[0, :]:
        for direction, nPanel in zip(['y', 'x'], axs.shape):
            if nPanel < 6:
                eval(f"ax.set_{direction}scale('log')")
                eval(f"ax.{direction}axis").set_minor_formatter(sensible_f)
            eval(f"ax.{direction}axis").set_major_formatter(sensible_f)

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
        for direction, nPanel in zip(['y', 'x'], axs.shape):
            if nPanel < 6:
                ax.grid(True, which="minor", axis=direction)

    # Not strictly compatible with gridspec height_ratios,
    # (throws warning), but still works ok.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        axs[0, 0].figure.tight_layout()
