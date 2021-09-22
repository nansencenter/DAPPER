"""Tools (notably `xpSpace`) for processing and presenting experiment data."""

import collections
import copy
import warnings

import colorama
import numpy as np
import struct_tools
from mpl_tools import place
from patlib.std import nonchalance
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
    so that the `dict` becomes a vector **space**.

    The coordinate system is specified by the `dims`:
    a list of keys defining the `namedtuple` of `self.Coord`.

    In intended usage, this space is highly sparse,
    meaning there are many coordinates with no entry.
    Indeed, as a data format for nd-arrays, it may be called
    "coordinate list representation", used e.g. by `scipy.sparse.coo_matrix`.

    Thus, operations across (potentially multiple) `dims`,
    such as optimization or averaging, should be carried out by iterating
    -- not over the `dims` -- but over the the list of items.

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

    This dict only has three `dims`, so this fails:
    >>> dct[(1, 2, 3, 4)]
    Traceback (most recent call last):
    ...
    KeyError: (1, 2, 3, 4)

    Individual coordinates can be anything. For example `None`:
    >>> dct[(1, 2, None)] = "point 2"
    """

    @property
    def dims(self):
        return self.Coord._fields

    def __init__(self, dims):
        """Usually initialized through `xpSpace`.

        Parameters
        ----------
        dims: list or tuple
            The attributes defining the coordinate system.
        """
        # Define coordinate system
        self.Coord = collections.namedtuple('Coord', dims)

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
                f"which has dims {self.dims}")
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
            outer = self.nest(outer_dims=list(key))  # nest
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
        coord = (getattr(entry, a, None) for a in self.dims)
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
        txt += " with Coord/dims: "
        try:
            txt += "(and ticks): " + str(struct_tools.AlignedDict(self.ticks))
        except AttributeError:
            txt += str(self.dims) + "\n"

        # Note: print(xpList(self)) produces a more human-readable table,
        # but requires prep_table(), which we don't really want to call again
        # (it's only called in from_list, not (necessarily) in any nested spaces)
        L = 2
        keys = [str(k) for k in self]
        if 2*L < len(keys):
            keys = keys[:L] + ["..."] + keys[-L:]
        keys = "[\n  " + ",\n  ".join(keys) + "\n]"
        return txt + f"populated by {len(self)} keys: {keys}"

    def nest(self, inner_dims=None, outer_dims=None):
        """Project along `inner_acces` to yield a new `xpSpace` with dims `outer_dims`

        The entries of this `xpSpace` are themselves `xpSpace`s, with dims `inner_dims`,
        each one regrouping the entries with the same (projected) coordinate.

        Note: is also called by `__getitem__(key)` if `key` is dict.
        """
        # Default: a singleton outer space,
        # with everything contained in the inner (projection) space.
        if inner_dims is None and outer_dims is None:
            outer_dims = ()

        # Validate dims
        if inner_dims is None:
            assert outer_dims is not None
            inner_dims = struct_tools.complement(self.dims, outer_dims)
        else:
            assert outer_dims is None
            outer_dims = struct_tools.complement(self.dims, inner_dims)

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

        This allows errors in the dims allotment, for ease-of-use.
        """
        absent = struct_tools.complement(attrs, self.dims)
        if absent:
            print(color_text("Warning:", colorama.Fore.RED),
                  "The requested attributes",
                  color_text(str(absent), colorama.Fore.RED),
                  ("were not found among the"
                   " xpSpace dims (attrs. used as coordinates"
                   " for the set of experiments)."
                   " This may be no problem if the attr. is redundant"
                   " for the coord-sys."
                   " However, if it is caused by confusion or mis-spelling,"
                   " then it is likely to cause mis-interpretation"
                   " of the shown results."))
            attrs = struct_tools.complement(attrs, absent)
        return attrs

    def append_dim(self, dims):
        self.__init__(self.dims+(dims,))
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
    """Functionality to facilitate working with `xps` and their results.

    `xpSpace.from_list` initializes a `SparseSpace` from a list
    of objects, typically experiments referred to as `xp`s, by

    - computing the relevant `dims` from the attributes, and
    - filling the dict by `xp`s.
    - computing and writing the attribute `ticks`.

    Using `xpSpace.from_list(xps)` creates a SparseSpace holding `xp`s.
    However, the nested `xpSpace`s output by `xpSpace.table_tree` will hold
    objects of type `UncertainQtty`,
    coz `xpSpace.table_tree` calls `mean` calls `get_stat(statkey)`.

    The main use of `xpSpace` is through `xpSpace.print` & `xpSpace.plot`,
    both of which call `xpSpace.table_tree` to nest the dims of the `SparseSpace`.
    """

    _ordering = dict(
        rot       = 'as_found',
        da_method = 'as_found',
    )

    @classmethod
    def from_list(cls, xps, ordering=None):
        """Init xpSpace from xpList."""

        def make_ticks(dims):
            """Unique & sort, for each dim (individually) in dims."""
            for name, values in dims.items():
                ticks = set(values)  # unique (jumbles order)
                order = {**cls._ordering, **(ordering or {})}
                order = order.get(name, 'default').lower()

                # Sort key
                if callable(order):
                    key = order
                elif 'as_found' in order:
                    key = values.index
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
                dims[name] = ticks

        # Define and fill SparseSpace
        xp_list = xpList(xps)
        dims = xp_list.prep_table(nomerge=['xSect'])[0]
        self = cls(dims)
        self.fill(xps)

        make_ticks(dims)
        # Note: this attr (ticks) will not be propagated through nest().
        # That is fine. Otherwise we should have to prune the ticks
        # (if they are to be useful), which we don't want to do.
        self.ticks = dims

        return self

    def fill(self, xps):
        """Mass insertion."""
        self.update([(self.coord_from_attrs(xp), xp) for xp in xps])

    def squeeze(self):
        """Eliminate unnecessary dimensions."""
        squeezed = xpSpace(xpList(self).prep_table()[0])
        squeezed.fill(self)
        return squeezed

    def get_stat(self, statkey="rmse.a"):
        """Make `xpSpace` with identical `keys`, but values `xp.avrgs.statkey`."""
        # Init a new xpDict to hold stat
        avrgs = self.__class__(self.dims)

        found_anything = False
        for coord, xp in self.items():
            val = getattr(xp.avrgs, statkey, None)
            avrgs[coord] = val
            found_anything = found_anything or (val is not None)

        if not found_anything:
            raise AttributeError(
                f"The stat.'{statkey}' was not found among any of the xp's.")

        return avrgs

    def mean(self, dims=None):
        # Note: The case `dims=()` should work w/o special treatment.
        if dims is None:
            return self

        nested = self.nest(dims)
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

    def tune(self, dims=None, costfun=None):
        """Get (compile/tabulate) a stat. optimised wrt. tuning params."""
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

    def validate_dims(self, dims):
        """Validate dims.

        Note: This does not convert None to (), allowing None to remain special.
              Use `()` if tuples are required.
        """
        roles = {}  # "inv"
        for role in set(dims) | set(DIM_ROLES):
            assert role in DIM_ROLES, f"Invalid role {role!r}"
            dd = dims.get(role, DIM_ROLES[role])

            if dd is None:
                pass  # Purposely special
            else:
                # Ensure iterable
                if isinstance(dd, str) or not hasattr(dd, "__iter__"):
                    dd = (dd,)

                dd = self.intersect_dims(dd)

                for dim in dd:

                    # Ensure unique
                    if dim in roles:
                        raise TypeError(
                            f"An dim (here {dim!r}) cannot be assigned to 2"
                            f" roles (here {role!r} and {roles[dim]!r}).")
                    else:
                        roles[dim] = role
            dims[role] = dd
        return dims

    def table_tree(self, statkey, dims, *, costfun=None):
        """Hierarchical nest(): xp_dict>outer>inner>mean>optim.

        as specified by `dims`. Returns this new xpSpace.

        - `print()` / `plot()` (respectively) separate
          tables    / panel(row)s for `dims['outer']`, and
          columns   / x-axis      for `dims['inner']`.

        - The `dims['mean']` and `dims['optim']` get eliminated
          by the mean/tune operations.

        Note: cannot support multiple statkeys because it's not (obviously) meaningful
              when optimizing over `dims['optim']`.
        """
        dims = self.validate_dims(dims)

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

        self = mean_tune(self)
        # Prefer calling mean_tune() [also see its docstring]
        # before doing outer/inner nesting. This is because then the dims of
        # a row (xpSpace) should not include mean&optim, and thus:
        #  - Column header/coords may be had directly as row.keys(),
        #    without extraction by coord_from_attrs() from (e.g.) row[0].
        #  - Don't need to propagate mean&optim dims down to the row level.
        #    which would require defining rows by the nesting:
        #    rows = table.nest(outer_dims=struct_tools.complement(table.dims,
        #        *(dims['inner'] or ()),
        #        *(dims['mean']  or ()),
        #        *(dims['optim'] or ()) ))
        #  - Each level of the output from table_tree
        #    is a smaller (and more manageable) dict.

        tables = self.nest(outer_dims=dims['outer'])
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

        return dims, tables

    def tickz(self, dim_name):
        """Dimension (axis) ticks without None"""
        return [x for x in self.ticks[dim_name] if x is not None]

    def print(self, statkey="rmse.a", dims=DIM_ROLES,  # noqa
              subcols=True, decimals=None, costfun=None,
              squeeze_labels=True, colorize=True):
        """Print tables of results.

        Parameters
        ----------
        statkey: str
            The statistic to extract from the `xp.avrgs` for each `xp`.
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
            It is acceptible to leave this empty, `mean=()` or `mean=None`.
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
        # Inform dims["mean"]
        if dims.get('mean', None):
            print(f"Averages (in time and) over {dims['mean']}.")
        else:
            print("Averages in time only"
                  " (=> the 1σ estimates may be unreliable).")

        dims, tables = self.table_tree(statkey, dims, costfun=costfun)

        def make_cols(rows, cc, subcols, h2):
            """Subcolumns: align, justify, join."""
            # Define subcol formats
            if subcols:
                templ = "{val} ±{prec}"
                templ += "" if dims['optim'] is None else " *{tuned_coord}"
                templ += "" if  dims['mean'] is None else " {nFail} {nSuccess}"  # noqa
                aligns = dict(prec="<", tuned_coord="<")
                labels = dict(val=statkey, prec="1σ",
                              tuned_coord=dims["optim"],
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

            # Get table's column coords/ticks (cc).
            # cc is really a set, but we use dict for ordering.
            # cc = self.ticks[dims["inner"]]  # may be > needed
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
            rows[0] = [h2+k for k in table.dims] + [h2+'⑊'] + rows[0]
            # Matter
            for i, (key, row) in enumerate(zip(table, rows[1:])):
                rows[i+1] = [*key] + ['|'] + row

            # Print
            print("\n", end="")
            if dims['outer']:
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

    def plot(self, statkey="rmse.a", dims=DIM_ROLES, get_style=default_styles,
             fignum=None, figsize=None, panels=None,
             title2=None, costfun=None, unique_labels=True,
             squeeze_labels=True):
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
            for a, panel in zip(dims["optim"], panelcol[1:]):
                yy = [getattr(coord, a, None) for coord in argmins]
                row.tuned_coords[a] = yy

                # Plotting all None's sets axes units (like any plotting call)
                # which can cause trouble if the axes units were actually supposed
                # to be categorical (eg upd_a), but this is only revealed later.
                if not all(y == None for y in yy):
                    row.handles[a] = panel.plot(xticks, yy, **style)

        def label_management():
            def manager(style):
                label = style.get("label", None)
                if unique_labels:
                    if label in register:
                        del style["label"]
                    else:
                        register.add(style["label"])
                        manager.has_labels = True
                elif label:
                    manager.has_labels = True
            manager.has_labels = False
            return manager
        register = set()  # mv inside to get legend on each panel

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
        assert len(dims["inner"]) == 1, "You must chose the abscissa."
        dims, tables = self.table_tree(statkey, dims, costfun=costfun)
        xticks = self.tickz(dims["inner"][0])

        # Create figure panels
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
        if title2 is not None:
            with nonchalance():
                title2 = title2.relative_to(rc.dirs["data"])
            fig_title += "\n" + str(title2)
        fig.suptitle(fig_title)

        # Loop outer
        for table_panels, (table_coord, table) in zip(panels.T, tables.items()):
            table.panels = table_panels

            label_manager = label_management()
            aa = xpList(table.keys()).prep_table()[0] if squeeze_labels else table.dims

            # Plot
            for coord, row in table.items():
                coord = NoneDict(struct_tools.intersect(coord._asdict(), aa))
                style = get_style(coord)
                label_manager(style)
                plot1(table.panels, row, style)

            beautify(table.panels,
                     title="" if dims["outer"] is None else table_coord.repr2(),
                     has_labels=label_manager.has_labels)

        tables.fig = fig
        tables.xp_dict = self
        tables.DIM_ROLES = dims
        return tables
