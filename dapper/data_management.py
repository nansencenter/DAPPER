"""Define xpSpace (subclasses SparseSpace (subclasses dict)),

which is handles the **presentation** of experiment (xp) results."""

##
from dapper import *

from collections import namedtuple
import hashlib
from matplotlib import ticker
import logging
mpl_logger = logging.getLogger('matplotlib')

##
def load_xps(savepath):
    """Load xp's (as list) from an .xps file or all .xp files in a dir.

    Note: saving this list as a new file (takes considerable time and)
          does not yield lower loading times."""

    savepath = os.path.expanduser(savepath)

    # SINGLE-HOST RUN, NEW FORMAT
    if savepath.endswith(".xps"):
        with open(savepath, "rb") as F:
            xps = dill.load(F)['xps']

    # SINGLE-HOST RUN, OLD FORMAT
    elif savepath.endswith(".pickle"):
        with open(savepath, "rb") as F:
            xps = dill.load(F)['xps']

    # PARALLELIZED (MP/GCP) RUN
    elif os.path.isdir(savepath):
        # savepath = savepath.rstrip(os.sep) # TODO: necessary?

        def load_xp(path):
            with open(path, "rb") as F:
                return dill.load(F)['xp']

        files = []
        for f in sorted_human(os.listdir(savepath)):
            f = os.path.join(savepath,f)
            if f.endswith("xp"): # MP RUN
            # if f.startswith("ixp_") and f.endswith(".xp"):
                files.append(f)
            elif os.path.isdir(f): # GCP RUN
                files.append(os.path.join(f,"xp"))

        print("Loading %d files from %s"%(len(files),savepath))
        # Dont use list comprehension (coz then progbar won't clean up correctly)
        xps = []
        for f in progbar(files,desc="Loading"):
            xps.append( load_xp(f) )

    else:
        raise RuntimeError("Could not locate xp(s) files")

    return xps

##
class SparseSpace(dict):
    """Dict, subclassed to make keys conform to a coord. system, i.e. a space.

    As for any dict, it can hold any type of objects.
    The coordinate system is specified by its "axes",
    which is used to produce self.Coord (a namedtuple class).

    Normally, this space is highly sparse,
    coz there are many coordinates with no matching experiment,
    eg. coord(da_method=Climatology, rot=True, ...).

    Indeed, operations across (potentially multiple simultaneous) axes,
    such as optimization or averaging,
    are internally carried out by iterating -- not across the axis --
    but across the the list of items, whose coordinates (dict keys)
    provide the axes ticks.

    The most important method is ``nest()``,
    which is used (by xpSpace.table_tree) to separate tables/columns,
    and also to carry out the mean/optim operations.

    Inspired by stackoverflow.com/a/7728830
    Also see stackoverflow.com/q/3387691
    """

    @property
    def axes(self):
        return self.Coord._fields

    def __init__(self, axes, *args, **kwargs):
        # Define coordinate class (i.e. system)
        self.Coord = namedtuple('Coord', axes) 

        # Optional: add repr
        self.Coord.__repr__ = lambda c: ",".join(f"{k}={v!r}" for k,v in zip(c._fields,c))
        self.Coord.__str__  = lambda c: ",".join(str(v) for v in c)

        # Write dict.
        # Use update() [not super().__init__] to pass by __setitem__(). 
        # Also see stackoverflow.com/a/2588648 & stackoverflow.com/a/2390997
        self.update(*args, **kwargs)

    def update(self, *args, **kwargs):
        """As for dict, but using __setitem__()."""
        for k, v in dict(*args, **kwargs).items(): self[k] = v

    def __setitem__(self, key, val):
        """Setitem ensuring coordinate conforms."""
        try: key = self.Coord(*key)
        except TypeError as err: raise TypeError(
                f"The key {key!r} did not fit the coord. system "
                f"which has axes {self.axes}")
        super().__setitem__(key,val)

    def __getitem__(self,key):
        """Indexing.
        
        In keeping with the mantra 'loop over points as list,
        not along actual axes' followed by nest(), this function
        also supports list indexing."""
        if isinstance(key,list): # Get list of items
            return [self[k] for k in key]
        elif isinstance(key, int) or isinstance(key, slice): # Slice
            return [*self.values()][key]
        else:
            # Get a single item by its coordinates
            # NB: Shouldn't use isinstance(key, self.Coord)
            # coz it fails when the namedtuple (Coord) has been
            # instantiated in different places (but with equal params).
            # Also see bugs.python.org/issue7796
            return super().__getitem__(key)

    def __repr__(self):
        s = self.__class__.__name__
        s += f"(axes={self.axes!r})"
        L = len(self)
        # Key list:
        n = min(L//2,2)
        head = [*self][:n]
        tail = [*self][n:]
        if len(tail) > len(head)+1:
            tail = tail[-(n+1):]
            tail[0] = "..."
        Open = "\n  Keys: ["
        sep  = ",\n" + " "*(len(Open)-1)
        keys = [str(x) for x in head+tail]
        keys = sep.join(keys)
        s += Open + keys + "]"
        # Length
        s += f"\n  Length: {L}"
        return s

    def matching_coords(self, **kwargs):
        # Get all items with attrs matching dict
        # Note: implmentation is simpler than for xpList.
        match  = lambda x: all(getattr(x,k)==kwargs[k] for k in kwargs)
        coords = [x for x in self if match(x)]
        return coords

    def get_coord(self,entry):
        """Inverse of (a purist version of) __getitem__."""
        coord = (getattr(entry,a,None) for a in self.axes)
        return self.Coord(*coord)

    def nest(self, inner_axes=None, outer_axes=None):
        """Return a new xpSpace with axes `outer_axes`,
        
        obtained by projecting along the ``inner_axes``.
        The entries of this xpSpace are themselves xpSpace's,
        with axes `inner_axes`,
        each one regrouping the entries with the same (projected) coordinate. 
        """

        # Default: a singleton outer space,
        # with everything contained in the inner (projection) space.
        if inner_axes is None and outer_axes is None:
            outer_axes = ()

        # Validate axes
        if inner_axes is None:
            assert outer_axes is not None
            inner_axes = complement(self.axes, *outer_axes)
        else:
            assert outer_axes is None
            outer_axes = complement(self.axes, *inner_axes)

        # Fill spaces
        outer_space = self.__class__(outer_axes)
        for coord, entry in self.items():
            outer_coord = outer_space.get_coord(coord)
            try:
                inner_space = outer_space[outer_coord]
            except KeyError:
                inner_space = self.__class__(inner_axes)
                outer_space[outer_coord] = inner_space
            inner_space[inner_space.get_coord(coord)] = entry

        return outer_space

    def add_axis(self, axis):
        self.__init__(self.axes+(axis,))
        for coord in list(self):
            entry = self.pop(coord)
            self[coord + (None,)] = entry

    def label_xSection(self,label,*NoneAttrs,**sub_coord):
        """Insert duplicate entries for the cross section
        
        whose ``coord``s match ``sub_coord``,
        adding the attr ``xSect=label`` to their ``coord``.

        This distinguishes the entries in this fixed-affine subspace,
        preventing them from being gobbled up in ``nest()``.

        Optionally, ``NoneAttrs`` can be specified to
        avoid these getting plotted in tuning panels.
        """

        if "xSect" not in self.axes:
            self.add_axis('xSect')

        for coord in self.matching_coords(**sub_coord):
            entry = self[coord]
            entry = deepcopy(entry)
            coord = coord._replace(xSect=label)
            coord = coord._replace(**{a:None for a in NoneAttrs})
            self[coord] = entry

##

DEFAULT_ALLOTMENT = dict(
        outer=None,
        inner=None,
        mean=None,
        optim=None,
        )

class xpSpace(SparseSpace):
    """Extends SparseSpace with functionality to present results from xps.

    Includes:
    
    - from_list() factory function insert a list of xps.
    - table_tree(), which is used by 
        - print()
        - plot()
    """

    @classmethod
    def from_list(cls, xps):

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
                ticks = set(arr) # unique (jumbles order)

                # Sort
                order = ordering.get(ax_name,'default').lower()
                if hasattr(order,'__call__'): # eg. mylist.index
                    key = order
                elif 'as_found' in order:
                    ticks = sorted(ticks, key=arr.index)
                else: # default sorting, with None placed at the end
                    ticks = sorted(ticks, key= lambda x: (x is None, x))
                if any(x in order for x in ['rev','inv']):
                    ticks = ticks[::-1]
                axes[ax_name] = ticks
            return axes

        # Define axes
        xp_list = xpList(xps)
        axes = xp_list.split_attrs(nomerge=['xSect'])[0]
        ticks = make_ticks(axes)
        self = cls(axes.keys())

        # Note: this attr (ticks) will not be propagated through nest().
        # That is fine. Otherwise we should have to prune the ticks
        # (if they are to be useful), which we don't want to do.
        self.ticks = axes

        # Fill
        self.update({self.get_coord(xp):xp for xp in xps})

        return self

    def field(self,statkey="rmse.a"):
        """Extract statkey for each item in self.
        
        Embellishments:
            - de_abbrev
            - found_anything
        """
        found_anything = False

        sk = de_abbrev(statkey)
        get_field = lambda xp: deep_getattr(xp,f'avrgs.{sk}',None)

        avrgs = self.__class__(self.axes)

        for coord, xp in self.items():
            a = get_field(xp)
            avrgs[coord] = a

            found_anything = found_anything or (a is not None)
        if not found_anything: raise RuntimeError(
                f"The stat. field '{statkey}' was not found"
                " among any of the xp's.")
        return avrgs

    def mean(self, axes=None):

        # Note: The case ``axes=()`` should work w/o special treatment.
        if axes is None: return self

        nested = self.nest(axes)
        for coord, space in nested.items():

            vals = [uq.val for uq in space.values()]

            # Don't use nanmean! It would give false impressions.
            mu = np.mean(vals)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore",category=RuntimeWarning)
                # Don't print warnings caused by N=1.
                # It already correctly yield nan's.
                var = np.var(vals,ddof=1)

            N = len(vals)
            uq = UncertainQtty(mu, sqrt(var/N))
            uq.nTotal   = N
            uq.nFail    = N - np.isfinite(vals).sum()
            uq.nSuccess = N - uq.nFail

            nested[coord] = uq
        return nested

    def tune(self, axes=None, costfun=None):
        """Get (compile/tabulate) a stat field optimised wrt. tuning params."""

        # Define cost-function
        costfun = (costfun or 'increasing').lower()
        if   'increas' in costfun: costfun = (lambda x: +x)
        elif 'decreas' in costfun: costfun = (lambda x: -x)
        else: assert hasattr(costfun, '__call__') # custom

        # Note: The case ``axes=()`` should work w/o special treatment.
        if axes is None: return self

        nested = self.nest(axes)
        for coord, space in nested.items():

            # Find optimal value and coord within space
            for i, (inner_coord, uq) in enumerate(space.items()):
                cost = costfun(uq.val)

                if i==0 or cost<=MIN:
                    MIN = cost
                    uq_opt = uq
                    uq_opt.tuning_coord = inner_coord

            nested[coord] = uq_opt

        return nested

    def validate_axes(self, axes):
        """Validate axes.

        Note: This does not convert None to (),
              allowing None to remain special. 
              Use ``axis or ()`` wherever tuples are required.
        """
        roles = {} # "inv"
        for role in set(axes) | set(DEFAULT_ALLOTMENT):
            assert role in DEFAULT_ALLOTMENT, f"Invalid role {role!r}"
            aa = axes.get(role,DEFAULT_ALLOTMENT[role])

            if aa is None:
                pass # Purposely special
            else:
                # Ensure iterable
                if isinstance(aa,str) or not hasattr(aa,"__iter__"):
                    aa = (aa,)

                for axis in aa:
                    # Ensure valid axis name
                    assert axis in self.axes, f"Axis {axis!r} not among the xp axes."
                    # Ensure unique
                    if axis in roles:
                        raise TypeError(f"An axis (here {axis!r}) cannot be assigned"
                        f" to 2 roles (here {role!r} and {roles[axis]!r}).")
                    else:
                        roles[axis] = role
            axes[role] = aa
        return axes

    
    def table_tree(xp_dict, statkey, axes):
        """Hierarchical nest(): xp_dict>outer>inner>mean>optim.

        as specified by ``axes``. Returns this new xpSpace.

        - print_1d / plot_1d (respectively) separate
          tables / panel(row)s for ``axes['outer']``, and
          columns/ x-axis      for ``axes['inner']``.

        - The ``axes['mean']`` and ``axes['optim']`` get eliminated
          by the mean()/tune() operations.

        Note: cannot support multiple statkeys
              because it's not (obviously) meaningful
              when optimizing over tuning_axes.
        """
        axes = xp_dict.validate_axes(axes)

        def mean_tune(xp_dict):
            """Take mean, then tune.

            Note: the SparseDict implementation should be sufficiently
            "uncluttered" that mean_tune() (or a few of its code lines)
            could be called anywhere above/between/below
            the ``nest()``ing of ``outer`` or ``inner``.
            These possibile call locations are commented in the code.
            """
            uq_dict = xp_dict.field(statkey)
            uq_dict = uq_dict.mean(axes['mean'])
            uq_dict = uq_dict.tune(axes['optim'])
            return uq_dict

        # Prefer calling mean_tune() [see its docstring] here:
        xp_dict = mean_tune(xp_dict)
        # This preference is because then the axes of
        # a row (xpSpace) should not include mean&optim, and thus:
        #  - Column header/coords may be had directly as row.keys(),
        #    without extraction by get_coord() from (e.g.) row[0].
        #  - Don't need to propagate mean&optim axes down to the row level.
        #    which would require defining rows by the nesting:
        #    rows = table.nest(outer_axes=complement(table.axes,
        #        *(axes['inner'] or ()),
        #        *(axes['mean']  or ()),
        #        *(axes['optim'] or ()) ))
        #  - Each level of the output from table_tree
        #    is a smaller (and more manageable) dict.

        tables = xp_dict.nest(outer_axes=axes['outer'])
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


    def axis_ticks_nn(self,axis_name):
        """Axis ticks without None"""
        return [x for x in self.ticks[axis_name] if x is not None]

    def print(xp_dict, statkey="rmse.a", axes=DEFAULT_ALLOTMENT, subcols=True):
        """Print table of results.

        - statkey: The statistical field from the experiments to report.

        - subcols: If True, then subcolumns are added to indicate the
                   1σ confidence interval, and potentially some other stuff.

        - axes: Allots (maps) each role to a set of axis of the xp_dict.
          Suggestion: dict(outer='da_method', inner='N', mean='seed', optim=('infl','loc_rad'))

          Example: If ``mean`` is assigned to:

          - ("seed",): Experiments are averaged accross seeds,
                       and the 1σ (sub)col is computed as sqrt(var(xps)/N),
                       where xps is a set of experiments.

          - ()       : Experiments are averaged across nothing (i.e. this is an edge case).

          - None     : Experiments are not averaged (i.e. the values are the same as above),
                       and the 1σ (sub)col is computed from the time series of that single experiment.
        """

        import pandas as pd

        # Enable storing None's (not as nan's) both in
        # - structured np.array
        # - pd.DataFrame
        _otype = object # could also use 'O'

        def align_subcols(rows,cc,subcols,h2):
            """Subcolumns: align, justify, join."""

            def unpack_uqs(uq_list, decimals=None):
                subcols=["val","conf","nFail","nSuccess","tuning_coord"]

                # np.array with named columns.
                dtype = np.dtype([(c,_otype) for c in subcols])
                avrgs = np.full_like(uq_list, dtype=dtype, fill_value=None)

                for i,uq in enumerate(uq_list):
                    if uq is not None:

                        # Format v,c
                        if decimals is None: v,c = uq.round(mult=0.2)
                        else:                v,c = np.round([uq.val, uq.conf],decimals)

                        # Write attr's
                        with set_tmp(uq,'val',v), set_tmp(uq,'conf',c):
                            for a in subcols:
                                try:
                                    avrgs[a][i] = getattr(uq,a)
                                except AttributeError:
                                    pass

                return avrgs

            columns = [list(x) for x in zip(*rows)] # Tranpose
            # Iterate over columns.
            for j, (col_coord, column) in enumerate(zip(cc, columns)):
                column = unpack_uqs(column)
                # Tabulate (sub)columns
                if subcols:

                    subc = dict()
                    subc['keys']     = ["val"   , "conf"]
                    subc['headers']  = [statkey , '1σ']
                    subc['frmts']    = [None    , None]
                    subc['spaces']   = [' ±'    , ] # last one gets appended below.
                    subc['aligns']   = ['>'     , '<'] # 4 header -- matter gets decimal-aligned.
                    if axes['optim'] is not None:
                        subc['keys']    += ["tuning_coord"]
                        subc['headers'] += [axes['optim']]
                        subc['frmts']   += [lambda x: tuple(a for a in x)]
                        subc['spaces']  += [' *']
                        subc['aligns']  += ['<']
                    elif axes['mean']  is not None:
                        subc['keys']    += ["nFail" , "nSuccess"]
                        subc['headers'] += ['☠'     , '✓'] # use width-1 symbols!
                        subc['frmts']   += [None    , None]
                        subc['spaces']  += [' '     , ' ']
                        subc['aligns']  += ['>'     , '>']
                    subc['spaces'].append('') # no space after last subcol
                    template = '{}' + '{}'.join(subc['spaces'])

                    # Tabulate subcolumns
                    subheaders = []
                    for key, header, frmt, _, align in zip(*subc.values()):
                        column[key] = tabulate_column(column[key],header,'æ',frmt=frmt)[1:]

                        h = str(header)
                        L = len(column[-1][key])
                        if align=='<': subheaders += [h.ljust(L)]
                        else:          subheaders += [h.rjust(L)]

                    # Join subcolumns:
                    matter = [template.format(*[row[k] for k in subc['keys']]) for row in column]
                    header = template.format(*subheaders)
                else:
                    column = column["val"]
                    column = tabulate_column(column,statkey,'æ')
                    header, matter = column[0], column[1:]

                if h2: # Do super_header
                    if j: super_header = str(col_coord)
                    else: super_header = repr(col_coord)
                    width = len(header) #+= 1 if using unicode chars like ✔️
                    super_header = super_header.center(width,"_")
                    header = super_header + "\n" + header

                columns[j] = [header]+matter
            # Un-transpose
            rows = [list(x) for x in zip(*columns)] 

            return rows

        axes, tables = xp_dict.table_tree(statkey, axes)
        for table_coord, table in tables.items():

            # Get this table's column coords (cc). Use dict for sorted&unique.
            # cc = xp_dict.ticks[axes["inner"]] # May be larger than needed.
            # cc = table[0].keys()              # May be too small a set.
            cc = {c:None for row in table.values() for c in row}

            # Convert table (rows) into rows (lists) of equal length
            rows = [[row.get(c,None) for c in cc] for row in table.values()]

            if False: # ****************** Simple (for debugging) table
                for i, (row_coord, row) in enumerate(zip(table, rows)):
                    row_key = ", ".join(str(v) for v in row_coord)
                    rows[i] =  [row_key]         + row
                rows.insert(0, [f"{table.axes}"] + [repr(c) for c in cc])
            else: # ********************** Elegant table.
                h2 = len(cc)>1 # do column-super-header
                rows = align_subcols(rows,cc,subcols,h2)

                # Make left_table and prepend.
                # - It's prettier if row_keys don't have unnecessary cols.
                # For example, the table of Climatology should not have an
                # entire column repeatedly displaying "infl=None".
                # => split_attrs().
                # - Why didn't we do this for the column attrs?
                # Coz there we have no ambition to split the attrs,
                # which would also require excessive processing:
                # nesting the table as cols, and then split_attrs() on cols.
                row_keys = xpList(table.keys()).split_attrs()[0]
                row_keys = pd.DataFrame.from_dict(row_keys,dtype=_otype)
                if len(row_keys.columns):
                    # Header
                    rows[0] = [('\n' if h2 else '')+k for k in row_keys] +\
                             [ '|' + ('\n|' if h2 else '')] + rows[0]
                    # Matter
                    for row, (i, key) in zip(rows[1:], row_keys.iterrows()):
                        rows[i+1] = [*key] + ['|']+ row

            # Print
            print("\n",end="")
            if axes['outer']:
                table_title = "•Table for " + repr(table_coord) + "."
                table_title = table_title + (f" •Averages Σ over {axes['mean']}." if axes['mean'] else "")
                with coloring(termcolors['underline']):
                    print(table_title)
            headers, *rows = rows
            print(tabulate_orig.tabulate(rows,headers).replace('æ',' '))

    
    def plot(xp_dict, statkey="rmse.a",
            axes=DEFAULT_ALLOTMENT,
            attrs_that_must_affect_color=('da_method',),
            # style_dict generated from:
            linestyle_axis=None,  linestyle_in_legend=True,
               marker_axis=None,     marker_in_legend=True,
                alpha_axis=None,      alpha_in_legend=True,
                color_axis=None,      color_in_legend=True,
            #
            fignum=None,
            costfun=None, 
            ):
        """Plot the avrgs of ``statkey`` as a function of axis["inner"].
        
        Initially, mean/optimum comps are done for
        ``axis["mean"]``, ``axis["optim"]``.
        The argmins are plotted on smaller axes below the main plot.
        The experiments can (optional) also be grouped by ``axis["outer"]``,
        yielding a figure with columns of panels.

        Assign ``style_axis``,
        where ``style`` is a linestyle aspect such as (linestyle, marker, alpha).
        If used, ``color_axis`` sets the cmap to a sequential (rainbow) colorscheme,
        whose coloring depends only on that attribute.
        """

        def _format_label(label):
            lbl = ''
            for k, v in label.items():
               if flexcomp(k, 'da_method', 'xSect'):
                   lbl = lbl + f' {v}'
               else:
                   lbl = lbl + f' {collapse_str(k)}:{v}'
            return lbl[1:]

        def _get_tick_index(coord,axis_name):
            tick = getattr(coord,axis_name)
            if tick is None:
                # By design, None should occur at end of axis,
                # and the index -1 would typically be a suitable flag.
                # However, sometimes we'd like further differentiation, and so:
                index = None
            else:
                ax = xp_dict.axis_ticks_nn(axis_name)
                index = ax.index(tick)
                if index == len(ax)-1:
                    index = -1
            return index

        from matplotlib.lines import Line2D
        markers = complement(Line2D.markers.keys(), ',')
        markers = markers[markers.index(".")+1:markers.index("_")]
        linestyles = ['--', '-.', ':']
        cmap = plt.get_cmap('jet')

        def _marker(index):
            axis = xp_dict.axis_ticks_nn(marker_axis)
            if index in [None, -1]:   return '.'
            else:                     return markers[index%len(markers)]
        def _linestyle(index):
            axis = xp_dict.axis_ticks_nn(linestyle_axis)
            if index in [None, -1]:   return '-'
            else:                     return linestyles[index%len(linestyles)]
        def _alpha(index):
            axis = xp_dict.axis_ticks_nn(alpha_axis)
            if   index in [None, -1]: return 1
            else:                     return ((1+index)/len(axis))**1.5
        def _color(index):
            axis = xp_dict.axis_ticks_nn(color_axis)
            if   index is None:       return None
            elif index is -1:         return cmap(1)
            else:                     return cmap((1+index)/len(axis))
        def _color_by_hash(x):
            """Color as a (deterministic) function of x."""

            # Particular cases
            if x=={'da_method': 'Climatology'}:
                return (0,0,0)
            elif x=={'da_method': 'OptInterp'}:
                return (0.5,0.5,0.5)
            else:
                # General case
                x = str(x).encode() # hashable
                # HASH = hash(tuple(x)) # Changes randomly each session
                HASH = int(hashlib.sha1(x).hexdigest(),16)
                colors = plt.get_cmap('tab20').colors
                return colors[HASH%len(colors)]

        # Style axes
        # Load kwargs into dict-of-dicts
        _eval = lambda s, ns=locals(): eval(s,None,ns)
        style_dict = {}
        for a in ['alpha','color','marker','linestyle']:
            if _eval(f"{a}_axis"):
                style_dict[a] = dict(
                    axis      = _eval(f"{a}_axis"),
                    in_legend = _eval(f"{a}_in_legend"),
                    formtr    = _eval(f"_{a}"),
                    )
        def styles_by_attr(attr):
            return [p for p in style_dict.values() if p['axis']==attr]
        styled_attrs = [p['axis'] for p in style_dict.values()]

        # Main axes
        axes, tables = xp_dict.table_tree(statkey, axes)
        xticks = xp_dict.axis_ticks_nn(axes["inner"][0])

        # Validate axes
        assert len(axes["inner"])==1, "You must chose the abscissa."
        for ak in style_dict:
            av = style_dict[ak]['axis']
            assert av in xp_dict.axes, f"Axis {av!r} not among xp_dict.axes."
            for bk in axes:
                bv = axes[bk]
                assert bv is None or (av not in bv), \
                        f"{ak}_axis={av!r} already used by axes[{bk!r}]"

        def get_style(coord):
            """Define line properties"""

            dct = {'markersize': 6}

            # Convert coord to label (dict with relevant attrs)
            label = {attr:val for attr,val in coord._asdict().items()
                    if ( (axes["outer"] is None) or (attr not in axes["outer"]) )
                    and val not in [None, "NULL", 'on x-axis']}

            # Assign color by label
            label1 = {attr:val for attr,val in label.items()
                     if attr in attrs_that_must_affect_color
                     or attr not in styled_attrs}
            dct['color'] = _color_by_hash(label1)

            # Assign legend label
            label2 = {attr:val for attr,val in label.items()
                     if attr not in styled_attrs
                     or any(p['in_legend'] for p in styles_by_attr(attr))}
            dct['label'] = _format_label(label2)

            # Get tick inds
            tick_inds = {}
            for attr,val in coord._asdict().items():
                if styles_by_attr(attr):
                    tick_inds[attr] = _get_tick_index(coord,attr)

            # Decide whether to label this line
            do_lbl = True
            for attr,val in coord._asdict().items():
                styles = styles_by_attr(attr)
                if styles and not any(style['in_legend'] for style in styles):
                    style = styles[0]
                    # Check if val has a "particular" value
                    do_lbl = tick_inds[attr] in [None,-1] 
                    if not do_lbl: break

            # Rm duplicate labels
            if not do_lbl or dct['label'] in label_register:
                dct['label'] = None
            else:
                label_register.append(dct['label'])

            # Format each style aspect
            for aspect, style in style_dict.items():
                attr = style['axis']
                S = style['formtr'](tick_inds[attr])
                if S is not None: dct[aspect] = S

            return dct

        # Setup Figure
        nrows = len(axes['optim'] or ()) + 1
        ncols = len(tables)
        screenwidth = 12.7 # my mac
        tables.fig, panels = freshfig(num=fignum, figsize=(min(5*ncols,screenwidth),7),
                nrows=nrows, sharex=True,
                ncols=ncols, sharey='row',
                gridspec_kw=dict(height_ratios=[6]+[1]*(nrows-1),hspace=0.05,
                    left=0.14, right=0.95, bottom=0.06, top=0.9))
        panels = np.ravel(panels).reshape((-1,ncols)) # atleast_2d (and col vectors)

        # Title
        ST = "Average wrt. time."
        if axes["mean"] is not None:
            ST = ST[:-1] + f" and {axes['mean']}."
        if style_dict:
            props = ", ".join(f"{a}:%s"%style_dict[a]['axis'] for a in style_dict)
            ST = ST + "\nProperty allotment: " + props + "."
        tables.fig.suptitle(ST)

        def plot_line(row, panels):
            """sort, insert None's, handle constants."""

            row.is_constant = all(x == row.Coord(None) for x in row)

            if row.is_constant:
                # This row is indep. of the x-axis => Make flat
                uqs = [row[0]]*len(xticks)
                row.style['marker'] = None
                row.style['lw'] = mpl.rcParams['lines.linewidth']/2
                # row.style['ls'] = "--"

            else:
                def get_uq(x):
                    coord = row.Coord(x)
                    try:             return row[coord]
                    except KeyError: return None
                # Sort uq's. Insert None if x missing.
                uqs = [get_uq(x) for x in xticks]

            # Extract attrs
            row.vals = [getattr(uq,'val',None) for uq in uqs]

            # Plot
            row.handles = {'top_panel':
                    panels[0].plot(xticks, row.vals, **row.style)[0]}

            # Plot tuning
            if axes["optim"]:

                # Extract attrs
                argmins = [getattr(uq,'tuning_coord',None) for uq in uqs]

                # Unpack tuning_coords. Write to row.
                row.tuning_coords = {}
                row.tuning_coords = {axis: [getattr(coord,axis,None)
                    for coord in argmins] for axis in axes["optim"]}

                # 
                for a, ax in zip(axes["optim"], panels[1:]):
                    row.handles[a] = ax.plot(xticks, row.tuning_coords[a], **row.style)

        # Loop panels
        label_register = [] # mv inside loop to get legend on each panel
        for ip, (table_coord, table) in enumerate(tables.items()):
            title = '' if axes["outer"] is None else repr(table_coord)
            table.panels = panels[:,ip]

            # Plot
            for coord, row in table.items():
                row.style = get_style(coord)
                plot_line(row, table.panels)
                
            # Beautify top_panel
            top_panel = table.panels[0]
            top_panel.set_title(title)
            if top_panel.is_first_col(): top_panel.set_ylabel(statkey)
            with set_tmp(mpl_logger, 'level', 99):
                top_panel.legend() # ignores "no label" msg
            # xlabel
            table.panels[-1].set_xlabel(axes["inner"][0])
            # Beautify tuning axes
            for a, ax in zip(axes["optim"] or (), table.panels[1:]):
                axis = xp_dict.axis_ticks_nn(a)
                if isinstance(axis[0], bool):
                    ylims = 0, 1
                else:
                    ylims = axis[0], axis[-1]
                ax.set_ylim(*stretch(*ylims,1.02))
                if ax.is_first_col():
                    ax.set_ylabel(f"Optim.\n{a}")

        return tables

def beautify_fig_ex3(tabulated_data, savepath, xp_dict):
    """Beautify.

    These settings are particular to example_3,
    and do not generalize well.
    """
    
    # Add savepath to suptitle
    try:
        savepath = savepath
        ST = tabulated_data.fig._suptitle.get_text()
        ST = "\n".join([ST, os.path.basename(savepath)])
        tabulated_data.fig.suptitle(ST)
    except NameError:
        pass

    # Get axs as array
    axs = array([col.panels for col in tabulated_data.values()]).T

    # Beautify main panels (top row):
    sensible_f = ticker.FormatStrFormatter('%g')
    for ax in axs[0,:]:
        for direction, nPanel in zip(['y','x'], axs.shape):
            if nPanel<6:
                eval(f"ax.set_{direction}scale('log')")
                eval(f"ax.{direction}axis").set_minor_formatter(sensible_f)
            eval(f"ax.{direction}axis").set_major_formatter(sensible_f)
        if "rmse" in ax.get_ylabel():
            ax.set_ylim([0.15, 5])

    # Beautify all panels
    for ax in axs.ravel():
        for direction, nPanel in zip(['y','x'], axs.shape):
            if nPanel<6:
                ax.grid(True,which="minor",axis=direction)
        # Inflation tuning panel
        if not ax.is_first_row():
            for axis in ['infl','xB','xN']:
                if axis in ax.get_ylabel():
                    yy = xp_dict.axis_ticks_nn(axis)
                    axis_scale_by_array(ax, yy, "y")
##
