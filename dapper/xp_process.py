"""Tools (notably `xpSpace`) for processing and presenting experiment data."""

import collections
import copy
import warnings

import colorama
import numpy as np
from mpl_tools import place
from patlib.std import nonchalance
from struct_tools import AlignedDict, complement, intersect, transps
from tabulate import tabulate

from dapper.dpr_config import rc
from dapper.stats import align_col, unpack_uqs
from dapper.tools.colors import color_text, stripe
from dapper.tools.rounding import UncertainQtty
from dapper.tools.viz import NoneDict, default_styles
from dapper.xp_launch import xpList


class SparseSpace(dict):
    """Subclass of `dict` that enforces key conformity to a given `namedtuple`.

    Like a normal `dict`, it can hold any type of objects.
    But, since the keys must conform, they effectively follow a coordinate system,
    so that the `dict` becomes a vector **space**. Example:
    >>> dct = xpSpace(["x", "y", "z"])
    >>> dct[(1, 2, 3)] = "pointA"

    The coordinate system is specified by the `dims`:
    a list of keys defining the `namedtuple` of `self.Coord`.
    The above dict only has three `dims`, so this fails:
    >>> dct[(1, 2, 3, 4)] = "pointB"  # doctest: +NORMALIZE_WHITESPACE
    Traceback (most recent call last):
    ...
    TypeError: The key (1, 2, 3, 4) did not fit the coord.  system
    which has dims ('x', 'y', 'z')

    Coordinates can contain any value, including `None`:
    >>> dct[(1, 2, None)] = "pointB"

    In intended usage, this space is highly sparse,
    meaning there are many coordinates with no entry.
    Indeed, as a data format for nd-arrays, it may be called
    "coordinate list representation", used e.g. by `scipy.sparse.coo_matrix`.

    Thus, operations across (potentially multiple) `dims`,
    such as optimization or averaging, should be carried out by iterating
    -- not over the `dims` -- but over the the list of items.

    The most important method is `nest`,
    which is used (by `xpSpace.table_tree`) to print and plot results.
    This is essentially a "groupby" operation, and indeed the case could
    be made that this class should be replaced by `pandas.DataFrame`.

    The `__getitem__` is quite flexible, allowing accessing by:

    - The actual key, a `self.Coord` object, or a standard tuple.<br>
      Returns single item. Example:

            >>> dct[1, 2, 3] == dct[(1, 2, 3)] == dct[dct.Coord(1, 2, 3)] == "pointA"
            True

    - A `slice` or `list`.<br>
      Returns list.<br>
      *PS: indexing by slice or list assumes that the dict is ordered,
      which we inherit from the builtin `dict` since Python 3.7.
      Moreover, it is a reflection of the fact that the internals of this class
      work by looping over items.*

    In addition, the `subspace` method (also aliased to `__call__`, and is implemented
    via `coords_matching`) can be used to select items by the values of a *subset*
    of their attributes. It returns a `SparseSpace`.
    If there is only a single item it can be accessed as in `dct[()]`.

    Inspired by

    - https://stackoverflow.com/a/7728830
    - https://stackoverflow.com/q/3387691
    """

    @property
    def dims(self):
        return self.Coord._fields

    def __init__(self, dims):
        """Usually initialized through `xpSpace.from_list`.

        Parameters
        ----------
        dims: list or tuple
            The attributes defining the coordinate system.
        """
        # Define coordinate system
        self.Coord = collections.namedtuple('Coord', dims)

        def repr2(c, keys=False, str_or_repr=repr):
            if keys:
                lst = [f"{k}={str_or_repr(v)}" for k, v in c._asdict().items()]
            else:
                lst = [str_or_repr(v) for v in c]
            return "(" + ", ".join(lst) + ")"

        self.Coord.repr2 = repr2

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
                f"which has dims {self.dims}")
        super().__setitem__(key, val)

    def __getitem__(self, key):
        """Also allows list-indexing by `list` and `slice`."""
        # List of items (from list of indices)
        if isinstance(key, list):
            lst = list(self.values())
            return [lst[k] for k in key]

        # List of items (from slice)
        elif isinstance(key, slice):
            return [*self.values()][key]

        # Single item (by Coord object, or tuple)
        else:
            # NB: Dont't use isinstance(key, self.Coord)
            # coz it fails when the namedtuple (Coord) has been
            # instantiated in different places (but with equal params).
            # Also see bugs.python.org/issue7796
            return super().__getitem__(key)

    def __call__(self, **kwargs):
        """Shortcut (syntactic sugar) for `SparseSpace.subspace`."""
        return self.subspace(**kwargs)

    def subspace(self, **kwargs):
        """Get an affine subspace.

        NB: If you're calling this repeatedly (for all values of the same `kwargs`)
        then you should consider using `SparseSpace.nest` instead.

        Example
        -------
            xp_dict.subspace(da_method="EnKF", infl=1, seed=3)
        """
        # Slow version
        # outer = self.nest(outer_dims=list(kwargs))  # make subspaceS
        # inner = outer[outer.Coord(**kwargs)]        # discard all but 1

        coords = self.coords_matching(**kwargs)
        inner = self.__class__(complement(self.dims, kwargs))
        for coord in coords:
            inner[inner.coord_from_attrs(coord)] = self[coord]

        return inner

    def coords_matching(self, **kwargs):
        """Get all `coord`s matching kwargs.

        Used by `SparseSpace.label_xSection` and `SparseSpace.subspace`. Unlike the
        latter, this function returns a *list* of *keys* of the *original subspace*.

        Note that the `missingval` shenanigans of `xpList.inds` are here unnecessary
        since each coordinate is complete.
        """
        def match(coord):
            return all(getattr(coord, k) == kwargs[k] for k in kwargs)

        return [c for c in self if match(c)]

    def coord_from_attrs(self, obj):
        """Form a `coord` for this `xpSpace` by extracting attrs. from `obj`.

        For instances of `self.Coord`, this is the identity opeartor, i.e.

            self.coord_from_attrs(coord) == coord
        """
        coord = (getattr(obj, a, None) for a in self.dims)
        return self.Coord(*coord)

    def __repr__(self):
        txt  = f"<{self.__class__.__name__}>"
        txt += " with Coord/dims: "
        try:
            txt += "(and ticks): " + str(AlignedDict(self.ticks))
        except AttributeError:
            txt += str(self.dims) + "\n"

        # Note: print(xpList(self)) produces a more human-readable table,
        # but requires prep_table(), which we don't really want to call again
        # (it's only called in from_list, not (necessarily) in any nested spaces)
        L = 2
        keys = [k.repr2() for k in self]
        if 2*L < len(keys):
            keys = keys[:L] + ["..."] + keys[-L:]
        keys = "[\n  " + ",\n  ".join(keys) + "\n]"
        return txt + f"populated by {len(self)} items with keys: {keys}"

    def nest(self, inner_dims=None, outer_dims=None):
        """Project along `inner_acces` to yield a new `xpSpace` with dims `outer_dims`

        The entries of this `xpSpace` are themselves `xpSpace`s, with dims `inner_dims`,
        each one regrouping the entries with the same (projected) coordinate.

        Note: this method could also be called `groupby`.
        Note: this method is also called by `__getitem__(key)` if `key` is dict.
        """
        # Default: a singleton outer space,
        # with everything contained in the inner (projection) space.
        if inner_dims is None and outer_dims is None:
            outer_dims = ()

        # Validate dims
        if inner_dims is None:
            assert outer_dims is not None
            inner_dims = complement(self.dims, outer_dims)
        else:
            assert outer_dims is None
            outer_dims = complement(self.dims, inner_dims)

        # Fill spaces
        outer_space = self.__class__(outer_dims)
        for coord, entry in self.items():
            # Lookup subspace coord
            outer_coord = outer_space.coord_from_attrs(coord)
            try:
                # Get subspace
                inner_space = outer_space[outer_coord]
            except KeyError:
                # Create subspace, embed
                inner_space = self.__class__(inner_dims)
                outer_space[outer_coord] = inner_space
            # Add entry to subspace, similar to .fill()
            inner_space[inner_space.coord_from_attrs(coord)] = entry

        return outer_space

    def intersect_dims(self, attrs):
        """Rm those `a` in `attrs` that are not in `self.dims`.

        This enables sloppy `dims` allotment, for ease-of-use.
        """
        absent = complement(attrs, self.dims)
        if absent:
            print(color_text("Warning:", colorama.Fore.RED),
                  "The requested attributes",
                  color_text(str(absent), colorama.Fore.RED),
                  ("were not found among the xpSpace dims"
                   " (attrs. used as coordinates for the set of experiments)."
                   " This may be no prob. if the attrs are redundant for the coord-sys."
                   " However, if due to confusion or mis-spelling, then it is likely"
                   " to cause mis-interpretation of the shown results."))
            attrs = complement(attrs, absent)
        return attrs

    def append_dim(self, dim):
        """Expand `self.Coord` by `dim`. For each item, insert `None` in new dim."""
        self.__init__(self.dims+(dim,))
        for coord in list(self):
            entry = self.pop(coord)
            self[coord + (None,)] = entry

    def label_xSection(self, label, *NoneAttrs, **sub_coord):
        """Insert duplicate entries for the given cross-section.

        Works by adding the attr. `xSection` to the dims of `SparseSpace`,
        and setting it to `label` for entries matching `sub_coord`,
        reflecting the "constance/constraint/fixation" this represents.
        This distinguishes the entries in this fixed-affine subspace,
        preventing them from being gobbled up by the operations of `nest`.

        If you wish, you can specify the `NoneAttrs`,
        which are consequently set to None for the duplicated entries,
        preventing them from being shown in plot labels and tuning panels.
        """
        if "xSect" not in self.dims:
            self.append_dim('xSect')

        for coord in self.coords_matching(**self.intersect_dims(sub_coord)):
            entry = copy.deepcopy(self[coord])
            coord = coord._replace(xSect=label)
            coord = coord._replace(**{a: None for a in NoneAttrs})
            self[coord] = entry


DIM_ROLES = dict(outer=None, inner=None, mean=None, optim=None)


class xpSpace(SparseSpace):
    """Functionality to facilitate working with `xps` and their results."""

    @classmethod
    def from_list(cls, xps, tick_ordering=None):
        """Init. from a list of objects, typically experiments referred to as `xp`s.

        - Computes the relevant `dims` from the attributes, and
        - Fills the dict by `xp`s.
        - Computes and writes the attribute `ticks`.

        This creates a `SparseSpace` of `xp`s. However, the nested subspaces generated
        by `xpSpace.table_tree` (for printing and plotting) will hold objects of type
        `UncertainQtty`, because it calls `mean` which calls `get_stat(statkey)`.
        """
        # Define and fill SparseSpace
        dct = xpList(xps).prep_table(nomerge=['xSect'])[0]
        self = cls(dct.keys())
        self.fill(xps)
        self.make_ticks(dct, tick_ordering)
        return self

    def make_ticks(self, dct, ordering=None):
        """Unique & sort, for each individual "dim" in `dct`. Assign to `self.ticks`.

        NB: `self.ticks` will not "propagate" through `SparseSpace.nest` or the like.
        """
        self.ticks = dct
        ordering = ordering or {}
        for name, values in dct.items():
            ticks = set(values)  # unique (jumbles order)
            order = ordering.get(name, 'as-found')

            # Sort key
            if callable(order):
                key = order
            elif 'as-found' in order:
                key = values.index
            else:  # "natural"
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
            dct[name] = ticks

    def fill(self, xps):
        """Mass insertion."""
        self.update([(self.coord_from_attrs(xp), xp) for xp in xps])

    def squeeze(self):
        """Eliminate unnecessary dimensions."""
        squeezed = xpSpace(xpList(self).prep_table()[0])
        squeezed.fill(self)
        return squeezed

    def get_stat(self, statkey):
        """Make `xpSpace` with same `Coord` as `self`, but values `xp.avrgs.statkey`."""
        # Init a new xpDict to hold stat
        avrgs = self.__class__(self.dims)

        not_found = set()
        for coord, xp in self.items():
            try:
                avrgs[coord] = getattr(xp.avrgs, statkey)
            except AttributeError:
                not_found.add(coord)

        if len(not_found) == len(self):
            raise AttributeError(
                f"The stat. '{statkey}' was not found among **any** of the xp's.")
        elif not_found:
            print(color_text("Warning:", "RED"), f"no stat. '{statkey}' found for")
            print(*not_found, sep="\n")

        return avrgs

    def mean(self, dims=None):
        """Compute mean over `dims` (a list). Returns `xpSpace` without those `dims`."""
        # Note: The case `dims=()` should work w/o special treatment.
        if dims is None:
            return self

        nested = self.nest(dims)
        for coord, space in nested.items():

            def getval(uq):
                return uq.val if isinstance(uq, UncertainQtty) else uq
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

    def tune(self, dims=None, costfun=None):
        """Get (compile/tabulate) a stat. optimised wrt. tuning params (`dims`)."""
        # Define cost-function
        costfun = (costfun or 'increasing').lower()
        if 'increas' in costfun:
            costfun = (lambda x: +x)
        elif 'decreas' in costfun:
            costfun = (lambda x: -x)
        else:
            assert callable(costfun)  # custom

        # Note: The case `dims=()` should work w/o special treatment.
        if dims is None:
            return self

        nested = self.nest(dims)
        for coord, space in nested.items():
            # Find optimal value (and coord) within space
            MIN = np.inf
            found_any = False
            for inner_coord, uq in space.items():
                cost = costfun(uq.val)
                if cost <= MIN:
                    found_any          = True
                    MIN                = cost
                    uq_opt             = uq
                    uq_opt.tuned_coord = inner_coord

            if not found_any:
                uq_opt = uq  # one is as good as another
                nDim = range(len(space.Coord._fields))
                uq_opt.tuned_coord = space.Coord(*(None for _ in nDim))

            nested[coord] = uq_opt

        return nested

    def table_tree(self, statkey, dims, *, costfun=None):
        """Make hierarchy `outer > inner > mean > optim` using `SparseSpace.nest`.

        The dimension passed to `nest` (at each level) is specified by `dims`.
        The dimensions of `dims['mean']` and `dims['optim']` get eliminated
        by the mean/tune operations. The `dims['outer']` and `dims['inner']
        become the keys for the output hierarchy.

        .. note::
            cannot support multiple `statkey`s because it's not (obviously) meaningful
            when optimizing over `dims['optim']`.
        """
        def validate_dims(dims):
            """Validate dims."""
            role_register = {}
            new = {}
            for role in set(dims) | set(DIM_ROLES):
                assert role in DIM_ROLES, f"Invalid role {role!r}"
                dd = dims.get(role, DIM_ROLES[role])

                if dd is None:
                    # Don't convert None to (), allowing None to remain special.
                    pass

                else:
                    # Ensure iterable
                    if isinstance(dd, str) or not hasattr(dd, "__iter__"):
                        dd = (dd,)

                    # Keep relevant only
                    dd = self.intersect_dims(dd)

                    # Ensure each dim plays a single-role
                    for dim in dd:
                        if dim in role_register:
                            raise TypeError(
                                f"A dim (here {dim!r}) cannot be assigned to 2"
                                f" roles (here {role!r} and {role_register[dim]!r}).")
                        else:
                            role_register[dim] = role
                new[role] = dd
            return new

        def mean_tune(xp_dict):
            """Take mean, then tune.

            Note: the `SparseSpace` implementation should be sufficiently
            "uncluttered" that `mean_tune` (or a few of its code lines)
            could be called anywhere above/between/below
            the `nest`ing of `outer` or `inner`.
            These possibile call locations are commented in the code.
            """
            uq_dict = xp_dict.get_stat(statkey)
            uq_dict = uq_dict.mean(dims['mean'])
            uq_dict = uq_dict.tune(dims['optim'], costfun)
            return uq_dict

        dims = validate_dims(dims)
        self2 = mean_tune(self)
        # Prefer calling mean_tune() [also see its docstring]
        # before doing outer/inner nesting. This is because then the dims of
        # a row (xpSpace) should not include mean&optim, and thus:
        #  - Column header/coords may be had directly as row.keys(),
        #    without extraction by coord_from_attrs() from (e.g.) row[0].
        #  - Don't need to propagate mean&optim dims down to the row level.
        #    which would require defining rows by the nesting:
        #    rows = table.nest(outer_dims=complement(table.dims,
        #        *(dims['inner'] or ()),
        #        *(dims['mean']  or ()),
        #        *(dims['optim'] or ()) ))
        #  - Each level of the output from table_tree
        #    is a smaller (and more manageable) dict.

        tables = self2.nest(outer_dims=dims['outer'])
        for table_coord, table in tables.items():
            # table = mean_tune(table)

            # Should not be used (nesting as rows is more natural,
            # and is required for getting distinct/row_keys).
            # cols = table.nest(outer_dims=dims['inner'])

            rows = table.nest(inner_dims=dims['inner'] or ())

            # Overwrite table by its nesting as rows
            tables[table_coord] = rows

            # for row_coord, row in rows.items():
            # rows[row_coord] = mean_tune(row)

        args = dict(statkey=statkey, xp_dict=self, dims=dims)
        tables.created_with = args
        return dims, tables

    def tickz(self, dim_name):
        """Dimension (axis) ticks without None"""
        return [x for x in self.ticks[dim_name] if x is not None]

    def print(self, statkey, dims,  # noqa (shadowing builtin)
              subcols=True, decimals=None, costfun=None,
              squeeze_labels=True, colorize=True, title=None):
        """Print tables of results.

        Parameters
        ----------
        statkey: str
            The statistic to extract from the `xp.avrgs` for each `xp`.
            Examples: `"rmse.a"` (i.e. `"err.rms.a"`), `"rmse.ocean.a"`, `"duration"`.
        dims: dict
            Allots (maps) the dims of `xpSpace` to different roles in the tables.

            - The "role" `outer` should list the dims/attributes
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
            It is acceptible to leave this empty: `mean=()` or `mean=None`.
        subcols: bool
            If `True`, then subcolumns are added to indicate

            - `1σ`: the confidence interval. If `mean=None` is used, this simply reports
              the value `.prec` of the `statkey`, providing this is an `UncertainQtty`.
              Otherwise, it is computed as `sqrt(var(xps)/N)`,
              where `xps` is the set of statistic gathered over the `mean` dimensions.
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
        # Title
        if title is not None:
            if colorize:
                clrs = colorama.Back.LIGHTBLUE_EX, colorama.Fore.BLACK
                title = color_text(str(title), *clrs)
            print(title)

        # Inform dims["mean"]
        if dims.get('mean', None):
            print(f"Averages (in time and) over {dims['mean']}.")
        else:
            print("Averages in time only"
                  " (=> the 1σ estimates may be unreliable).")

        def make_cols(rows, cc, subcols, h2):
            """Subcolumns: align, justify, join."""
            # Define subcol formats
            if subcols:
                templ = "{val} ±{prec}"
                templ += "" if dims['optim'] is None else " *{tuned_coord}"
                templ += "" if dims['mean' ] is None else " {nFail} {nSuccess}"
                aligns = dict(prec="<", tuned_coord="<")

            def align(column, idx):
                if idx == 0:
                    headers = dict(val=statkey, prec="1σ", tuned_coord=dims["optim"])
                else:
                    headers = dict(val="", prec="1σ", tuned_coord="")
                headers.update(nFail="☠", nSuccess="✓")

                col = unpack_uqs(column, decimals)

                if subcols:
                    for key in list(col):
                        if key in templ:
                            subcolmn = [headers.get(key, key)] + col[key]
                            col[key] = align_col(subcolmn, just=aligns.get(key, ">"))
                        else:
                            del col[key]
                    col = [templ.format(**row) for row in transps(col)]
                else:
                    col = align_col([headers["val"]] + col["val"])
                return col

            def super_header(col_coord, idx, col):
                header, matter = col[0], col[1:]
                cc = col_coord.repr2(not idx, str).strip("()").replace(", ", ",")
                cc = cc.center(len(header), "_")  # +1 width for wide chars like ✔️
                return [cc + "\n" + header] + matter

            # Transpose
            columns = [list(x) for x in zip(*rows)]

            # Format column
            for j, (col_coord, column) in enumerate(zip(cc, columns)):
                col = align(column, j)
                if h2:
                    col = super_header(col_coord, j, col)
                columns[j] = col

            # Un-transpose
            rows = [list(x) for x in zip(*columns)]

            return rows

        dims, tables = self.table_tree(statkey, dims, costfun=costfun)

        for table_coord, table in tables.items():

            # Get table's column coords/ticks (cc).
            # cc is really a set, but we use dict for ordering.
            # cc = self.ticks[dims["inner"]]  # may be > needed
            # cc = table[0].keys()            # may be < needed
            cc = {c: None for row in table.values() for c in row}
            # Could additionally do cc = table.squeeze() but is it worth it?

            # Convert table (rows) into rows (lists) of equal length
            rows = [[row.get(c, None) for c in cc] for row in table.values()]

            # Align cols
            h2 = "\n" if len(cc) > 1 else ""  # super-header?
            headers, *rows = make_cols(rows, cc, subcols, h2)

            # Prepend left-side (attr) table
            if squeeze_labels:
                table = table.squeeze()
            headers = [h2+k for k in table.dims] + [h2+'⑊'] + headers
            for i, (key, row) in enumerate(zip(table, rows)):
                rows[i] = [*key] + ['|'] + row

            print()
            if dims['outer']:
                # Title
                table_title = "Table for " + table_coord.repr2(True).strip("()")
                if colorize:
                    clrs = colorama.Back.YELLOW, colorama.Fore.BLACK
                    table_title = color_text(table_title, *clrs)
                print(table_title)
            table = tabulate(rows, headers).replace('␣', ' ')
            if colorize:
                table = stripe(table, slice(2, None))
            print(table)

        return tables

    def plot(self, statkey, dims, get_style=default_styles,
             fignum=None, figsize=None, panels=None, costfun=None,
             title1=None, title2=None, unique_labels=True, squeeze_labels=True):
        """Plot (tables of) results.

        Analagously to `xpSpace.print`,
        the averages are grouped by `dims["inner"]`,
        which here plays the role of the x-axis.

        The averages can also be grouped by `dims["outer"]`,
        producing a figure with multiple (columns of) panels.

        The optimal points/parameters/attributes are plotted in smaller panels
        below the main plot. This can be turned off by providing the figure
        dims through the `panels` argument.

        The parameters `statkey`, `dims`, `costfun`, `sqeeze_labels`
        are documented in `xpSpace.print`.

        Parameters
        ----------
        get_style: function
            A function that takes an object, and returns a dict of line styles,
            usually as a function of the object's attributes.
        title1: anything
            Figure title (in addition to the the defaults).
        title2: anything
            Figure title (in addition to the defaults). Goes on a new line.
        unique_labels: bool
            Only show a given line label once, even if it appears in several panels.
        squeeze_labels:
            Don't include redundant attributes in the labels.
        """
        def plot1(panelcol, row, style):
            """Plot a given line (row) in the main panel and the optim panels.

            Involves: Sort, insert None's, handle constant lines.
            """
            # Make a full row (yy) of vals, whether is_constant or not.
            # is_constant = (len(row)==1 and next(iter(row))==row.Coord(None))
            is_constant = all(x == row.Coord(None) for x in row)
            if is_constant:
                yy = [row[None, ] for _ in xticks]
                style.marker = None
            else:
                yy = [row.get(row.Coord(x), None) for x in xticks]

            # Plot main
            row.vals = [getattr(y, 'val', None) for y in yy]
            row.handles = {}
            row.handles["main_panel"] = panelcol[0].plot(xticks, row.vals, **style)[0]

            # Plot tuning params
            row.tuned_coords = {}  # Store ordered, "transposed" argmins
            argmins = [getattr(y, 'tuned_coord', None) for y in yy]
            for a, panel in zip(dims["optim"] or (), panelcol[1:]):
                yy = [getattr(coord, a, None) for coord in argmins]
                row.tuned_coords[a] = yy

                # Plotting all None's sets axes units (like any plotting call)
                # which can cause trouble if the axes units were actually supposed
                # to be categorical (eg upd_a), but this is only revealed later.
                if not all(y == None for y in yy):
                    style["alpha"] = 0.2
                    row.handles[a] = panel.plot(xticks, yy, **style)

        def label_management(table):
            def pruner(style):
                label = style.get("label", None)
                if unique_labels:
                    if label in register:
                        del style["label"]
                    elif label:
                        register.add(style["label"])
                        pruner.has_labels = True
                elif label:
                    pruner.has_labels = True
            pruner.has_labels = False

            def squeezer(coord):
                return intersect(coord._asdict(), label_attrs)
            if squeeze_labels:
                label_attrs = xpList(table.keys()).prep_table()[0]
            else:
                label_attrs = table.dims

            return pruner, squeezer
        register = set()

        def beautify(panels, title, has_labels):
            panel0 = panels[0]
            # panel0.set_title(title)
            panel0.text(.5, 1, title, fontsize=12, ha="center", va="bottom",
                        transform=panel0.transAxes, bbox=dict(
                            facecolor='lightyellow', edgecolor='k',
                            alpha=0.99, boxstyle="round,pad=0.25",
                            # NB: padding makes label spill into axes
                        ))
            if has_labels:
                panel0.legend()
            if panel0.is_first_col():
                panel0.set_ylabel(statkey)
            panels[-1].set_xlabel(dims["inner"][0])
            # Tuning panels:
            for a, panel in zip(dims["optim"] or (), panels[1:]):
                if panel.is_first_col():
                    panel.set_ylabel(f"Optim.\n{a}")

        # Nest dims through table_tree()
        dims, tables = self.table_tree(statkey, dims, costfun=costfun)
        assert len(dims["inner"]) == 1, "You must chose a valid attr. for the abscissa."

        if not hasattr(self, "ticks"):
            # TODO 6: this is probationary.
            # In case self is actually a subspace, it may be that it does not contain
            # all of the ticks of the original xpSpace. This may be fine,
            # and we generate the ticks here again. However, this is costly-ish, so you
            # should maybe simply (manually) assign them from the original xpSpace.
            # And maybe you actually want the plotted lines to have holes where self
            # has no values. Changes in the ticks are not obvious to the naked eye,
            # unlike the case for printed tables (where column changes are quite clear).
            print(color_text("Warning:", colorama.Fore.RED), "Making new x-ticks."
                  "\nConsider assigning them yourself from the original"
                  " xpSpace to this subspace.")
            self.make_ticks(xpList(self).prep_table()[0])
        xticks = self.tickz(dims["inner"][0])

        # Create figure axes
        if panels is None:
            nrows   = len(dims['optim'] or ()) + 1
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
        if dims["mean"] is not None:
            fig_title += " and " + ", ".join([repr(c) for c in dims['mean']])
        if title1 is not None:
            fig_title += ". " + title1
        if title2 is not None:
            with nonchalance():
                title2 = title2.relative_to(rc.dirs["data"])
            fig_title += "\n" + str(title2)
        fig.suptitle(fig_title)

        # Loop outer
        for ax_column, (table_coord, table) in zip(panels.T, tables.items()):
            table.panels = ax_column
            label_prune, label_squeeze = label_management(table)
            for coord, row in table.items():
                style = get_style(NoneDict(label_squeeze(coord)))
                label_prune(style)
                plot1(table.panels, row, style)

            beautify(table.panels,
                     title=("" if dims["outer"] is None else
                            table_coord.repr2(True).strip("()")),
                     has_labels=label_prune.has_labels)

        tables.fig = fig  # add reference to fig
        return tables
