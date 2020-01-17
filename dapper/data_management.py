"""Define xpCube,

which is handles the **presentation** of experiment (xp) results."""

##
from dapper import *

##
from collections import namedtuple
import hashlib
from matplotlib import ticker
import logging
mpl_logger = logging.getLogger('matplotlib')


# TODO: rename xpCube, xpList? Also remember "hypercube" 
class ExperimentHypercube(NestedPrint):
    """View the list of xps as an n-rectangle whose dims correspond to attributes.

    This hyper-rectangle ("hypercube") is of shape: (len(ax) for ax in axes),
    where axes are the distinct attributes of xp_list.

    Normally, this array is highly sparse: array.size >> len(xps),
    coz there are many coordinates with no matching experiment,
    eg. Coord(da_method=Climatology, rot=True, ...).

    Therefore, the hypercube IS NOT EXPLICITly represented as an array,
    but kept as a dict whose keys are coordinates.

    Indeed, operations along an axis (optimization or averaging)
    are internally carried out by iterating -- not across the axis --
    but across the the list of xps (and allocating to the appropriate axis tick).

    Since "optimality" is only defined for a given field (eg avrg rmse.a),
    such operations are not done at initialization.
    """

    #----------------------------------
    # Core "hypercube" functionality
    #----------------------------------

    printopts = dict(excluded=['xp_list','xp_dict','Coord'])
    tags = (re.compile(r'^tag\d'), )

    def __init__(self, xp_list, axes=None):
        xp_list = ExperimentList(xp_list)

        # Define axes
        if axes is None:
            distinct_attrs = xp_list.split_attrs(nomerge=self.tags)[0]
            axes = distinct_attrs
        self.axes = axes
        self.make_ticks()
        # Define coord (Hashable, unlike dict. Fixed-length, unlike classes)
        self.Coord = namedtuple('Coord', self.axes)

        # Fill "hypercube"
        self.xp_list = xp_list
        self.xp_dict = {self.coord(xp): xp for xp in xp_list}

    def make_ticks(self, ordering=dict(
                N         = 'default',
                seed      = 'default',
                infl      = 'default',
                loc_rad   = 'default',
                rot       = 'as_found',
                da_method = 'as_found',
                )):
        """Make and order ticks of all axes."""

        for ax_name, arr in self.axes.items():
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
            self.axes[ax_name] = ticks

    def __getitem__(self,key):
        """Get items from self.xp_list"""
        if hasattr(key,'da_method'):
            # Get a single item by its coordinates
            return self.xp_dict[self.Coord(*key)]
        elif isinstance(key, dict):
            # Get all items with attrs matching dict
            match_attr = lambda xp, k: getattr(xp,k,None)==key[k]
            match_dict = lambda xp: all(match_attr(xp,k) for k in key)
            return [xp for xp in self.xp_list if match_dict(xp)]
        elif isinstance(key,list):
            # Get list of items
            return ExperimentList([self[k] for k in key])
        else:
            return self.xp_list[key]

    def coord(self,xp):
        """Inverse of __getitem__"""
        axes = self.axes
        coord = (getattr(xp,ax,None) for ax in axes)
        # To use indices rather than the values themselves:
        # coord = (axes[ax].index(getattr(xp,ax,None)) for ax in axes)
        return self.Coord(*coord)

    def group_along(self, *projection_axs, nullval="NULL"):
        """Group indices of xp_list by their coordinates,

        whose ticks along *projection_axs are set to nullval."""
        projection_axs = {ax:nullval for ax in projection_axs}
        groups = {}
        for ix, xp in enumerate(self.xp_list):
            coord = self.coord(xp)._replace(**projection_axs)
            if coord in groups:
                groups[coord].append(ix)
            else:
                groups[coord] = [ix]
        return groups

    #----------------------------------
    # Other functionality
    #----------------------------------
    def axis_ticks_nn(self,axis_name):
        """Axis ticks without None"""
        return [x for x in self.axes[axis_name] if x is not None]

    def single_out(self,coords,tag=None,NoneAttrs=()):
        """Insert duplicates of self[coords], with a tag.

        This is to distinguish them from all other xps,
        which prevents them being gobbled up in averaging/optimization."""
        xps = []
            
        for xp in self[coords]:
            xp = deepcopy(xp)
            xp.tag1 = tag

            # Avoid plotting optimal values.
            for a in NoneAttrs:
                setattr(xp,a,None)

            xps.append(xp)

        # "Manually" define axes coz split_attrs() is slow
        axes = {**self.axes, **dict(tag1=[tag,None]+self.axes.get('tag1',[]))}
        self.__init__(self.xp_list+xps, axes)

    #----------------------------------
    # Processing wrt. to a stat. field
    #----------------------------------
    def compile_avrgs(self,statkey="rmse.a"):
        """Baically [getattr(xp.avrgs,statkey) for xp in xp_list]"""
        statkey = de_abbrev(statkey)
        avrg = lambda xp: deep_getattr(xp,f'avrgs.{statkey}.val',None)
        return [avrg(xp) for xp in self.xp_list]

    def mean_field(self, statkey="rmse.a", axis=('seed',)):
        stats = np.asarray(self.compile_avrgs(statkey))
        groups = self.group_along(*axis)
        mean_cube = {}
        for coord,inds in groups.items():
            # Don't use nanmean! It would give false impressions.
            vals = stats[inds]
            N = len(vals)
            uq = UncertainQtty(np.mean(vals), sqrt(np.var(vals,ddof=1)/N))
            uq.nTotal   = N
            uq.nFail    = N - np.isfinite(vals).sum()
            uq.nSuccess = N - uq.nFail
            mean_cube[coord] = uq
        return mean_cube

    def tuned_field(self,
            statkey="rmse.a", costfun=None,
            mean_axs=('seed',), optim_axs=(),
            ):
        """Get (compile/tabulate) a stat field optimised wrt. tuning params."""

        # Define cost-function
        costfun = (costfun or 'increasing').lower()
        if   'increas' in costfun: costfun = (lambda x: +x)
        elif 'decreas' in costfun: costfun = (lambda x: -x)
        else: assert hasattr(costfun, '__call__') # custom

        # Average
        mean_cube = self.mean_field(statkey, mean_axs)
        mean_subspace = {ax:"NULL" for ax in mean_axs}

        # Gather along optim_axs (and mean_axs)
        tuned_mean_groups = self.group_along(*optim_axs,*mean_axs)

        optimum_cube = {}
        for group_coord, inds_in_group in tuned_mean_groups.items():

            # Find optimal value and coord within group
            optimum = np.inf, None
            for ix in inds_in_group:

                # Get value from mean_cube
                coord = self.coord(self[ix])
                mean_coord = coord._replace(**mean_subspace)
                # TODO: AFAICT, mean_coord gets re-produced and re-checked
                #       for all seeds, which is unnecessary.
                mean_val   = mean_cube[mean_coord].val

                cost = costfun(mean_val)
                if cost < optimum[0]: # always False for NaN's  
                    optimum = cost, mean_coord

            optimum_cube[group_coord] = optimum

        return optimum_cube



def load_xps(savepath):
    """Load xp's (as list) from an .xps file or all .xp files in a dir.

    Note: saving this list in a new file (takes considerable time and)
          does not yield lower loading times."""

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

    return xps




def plot1d(hypercube, x_ax, 
        statkey="rmse.a", costfun=None,
        mean_axs=(), optim_axs=(),
         panel_ax=None,
         # style_dict:
        linestyle_ax=None,        linestyle_in_legend=True,
           marker_ax=None,           marker_in_legend=True,
            alpha_ax=None,            alpha_in_legend=True,
            color_ax=None,            color_in_legend=True,
        #
        attrs_that_must_affect_color=('da_method',),
        fignum=None,
        ):
    """Plot the avrgs of ``statkey`` as a function of the attribute ``x_ax``.

    Firstly, mean/optimum comps are done for ``mean_axs``, ``optim_axs``.
    The argmins are plotted on smaller axes below the main plot.

    The experiments can (optional) also be distributed to a row of panels,
    one for each value of an attribute set in ``panel_ax``.

    The remaining attributes constitute the legend key for each plotted curve.
    They are also used to color each line differently.
    However, these attributes may be subtracted from by assigning ``style_ax``,
    where ``style`` is a linestyle aspect such as (linestyle, maker, alpha).
    If used, ``color_ax`` sets the cmap to a sequential (rainbow) colorscheme,
    whose coloring depends only on that attribute.
    """

    # Panel
    panels = hypercube.axes.get(panel_ax,[None])

    def _format_label(label):
        lbl = ''
        for k, v in label.items():
           if flexcomp(k, 'da_method', *hypercube.tags):
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
            ax = hypercube.axis_ticks_nn(axis_name)
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
        axis = hypercube.axis_ticks_nn(marker_ax)
        if index in [None, -1]:   return '.'
        else:                     return markers[index%len(axis)]
    def _linestyle(index):
        axis = hypercube.axis_ticks_nn(linestyle_ax)
        if index in [None, -1]:   return '-'
        else:                     return linestyles[index%len(axis)]
    def _alpha(index):
        axis = hypercube.axis_ticks_nn(alpha_ax)
        if   index in [None, -1]: return 1
        else:                     return ((1+index)/len(axis))**1.5
    def _color(index):
        axis = hypercube.axis_ticks_nn(color_ax)
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

    # Load Style info into dict-of-dicts
    _eval = lambda s, ns=locals(): eval(s,None,ns)
    style_dict = {}
    for a in ['alpha','color','marker','linestyle']:
        if _eval(f"{a}_ax"):
            style_dict[a] = dict(
                axis      = _eval(f"{a}_ax"),
                in_legend = _eval(f"{a}_in_legend"),
                formtr    = _eval(f"_{a}"),
                )
    def styles_by_attr(attr):
        return [p for p in style_dict.values() if p['axis']==attr]
    styled_attrs = [p['axis'] for p in style_dict.values()]

    def set_style(coord):
        """Define line properties"""

        dct = {'markersize': 6}

        # Convert coord to label (dict with relevant attrs)
        label = {attr:val for attr,val in coord._asdict().items()
                if attr is not panel_ax
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

    # Validate tuning axes
    xcld      = (*styled_attrs, panel_ax)
    optim_axs = intersect(complement(optim_axs, *xcld), *hypercube.axes)
    mean_axs  = intersect(complement(mean_axs , *xcld), *hypercube.axes)

    # Get tuned hypercube
    tuned_cube = hypercube.tuned_field(statkey, costfun, mean_axs, optim_axs)

    # Setup Figure
    nrows = len(optim_axs) + 1
    ncols = len(panels)
    screenwidth = 12.7 # my mac
    fig, axs = freshfig(num=fignum, figsize=(min(5*ncols,screenwidth),7),
            nrows=nrows, sharex=True,
            ncols=ncols, sharey='row',
            gridspec_kw=dict(height_ratios=[6]+[1]*(nrows-1),hspace=0.05,
                left=0.14, right=0.95, bottom=0.06, top=0.9))
    axs = np.ravel(axs).reshape((-1,ncols)) # atleast_2d (and col vectors)
    # Title
    ST = "Stats avrg'd in time."
    if mean_axs:
        ST = ST[:-1] + f" and wrt. {mean_axs}."
    if style_dict:
        props = ", ".join(f"{a}: %s"%style_dict[a]['axis'] for a in style_dict)
        ST = ST + " Line property attribution:\n" + props + "."
    fig.suptitle(ST)

    xticks = hypercube.axis_ticks_nn(x_ax)
    def arrays_from_dict(line,xticks):
        """Extract line data: from dict[coords] to array."""

        line['is_constant'] = all(x is None for x in line)

        # Extract yvalues, argmins
        if line['is_constant']:
            # This line is indep. of the x-axis => Make flat
            entry   = line.pop(None)
            yvalues = [entry[0]]*len(xticks)
            argmins = [entry[1]]*len(xticks)
        else:
            zipped = [line.pop(x,[None,None]) for x in xticks]
            yvalues, argmins = np.array(zipped).T
        line['stat_values'] = yvalues

        # getattr(argmin,optim_ax)
        line['tuning_values'] = {}
        for attr in optim_axs:
            ticks = [getattr(coord,attr,None) for coord in argmins]
            line['tuning_values'][attr] = ticks

    # Loop panels
    plot_data_per_column = []
    label_register = [] # mv inside to get legend on each panel
    for ip, (panel, top_ax) in enumerate(zip(panels, axs[0,:])):
        panel_lines = {}
        plot_data_per_column += [panel_lines]
        title = '' if panel_ax is None else f'{panel_ax}: {panel}'

        # Gather line data (iterating over xp coords)
        for coord in tuned_cube:
            if panel_ax is None or panel==getattr(coord,panel_ax):
                y = tuned_cube[coord]
                x = getattr(coord,x_ax)
                coord = coord._replace(**{x_ax:'on x-axis'})
                panel_lines.setdefault(coord,{})[x] = y

        # Order data
        for line in panel_lines.values():
            arrays_from_dict(line, xticks)

        # Plot
        for coord, line in panel_lines.items():

            # Set line properties
            ln_props = set_style(coord)
            if line['is_constant']:
                ln_props['marker'] = None
            
            # Plot
            top_ax.plot(xticks, line['stat_values'],         **ln_props)
            for attr, ax in zip(optim_axs, axs[1:,ip]):
                ax.plot(xticks, line['tuning_values'][attr], **ln_props)

        # Beautify main axes
        top_ax.set_title(title)
        top_ax.set_ylabel(statkey)
        with set_tmp(mpl_logger, 'level', 99):
            top_ax.legend() # ignores "no label" msg

        axs[-1,ip].set_xlabel(x_ax)

        # Beautify tuning axes
        for attr, ax in zip(optim_axs, axs[1:,ip]):
            axis = hypercube.axis_ticks_nn(attr)
            if isinstance(axis[0], bool):
                ylims = 0, 1
            else:
                ylims = axis[0], axis[-1]
            ax.set_ylim(*stretch(*ylims,1.02))
            if ax.is_first_col():
                ax.set_ylabel(f"Optim.\n{attr}")

        def adjust_axs(axs):
            """Beautify axes. Maybe belongs outside plot1d."""
            sensible_f = ticker.FormatStrFormatter('%g')
            # Main panels (top row):
            for ax in axs[0,:]:
                for direction, nPanel in zip(['y','x'], axs.shape):
                    if nPanel<6:
                        eval(f"ax.set_{direction}scale('log')")
                        eval(f"ax.{direction}axis").set_minor_formatter(sensible_f)
                    eval(f"ax.{direction}axis").set_major_formatter(sensible_f)
                if "rmse" in ax.get_ylabel():
                    ax.set_ylim([0.15, 5])
            # All panels
            for ax in axs.ravel():
                for direction, nPanel in zip(['y','x'], axs.shape):
                    if nPanel<6:
                        ax.grid(True,which="minor",axis=direction)
                # Inflation tuning panel
                if not ax.is_first_row() and 'infl' in ax.get_ylabel():
                    ticks = hypercube.axis_ticks_nn('infl')
                    axis_scale_by_array(ax, ticks, "y")
        adjust_axs(axs)

    return fig, axs, hypercube, plot_data_per_column
