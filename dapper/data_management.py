"""Define xpSpace,

which is handles the **presentation** of experiment (xp) results."""

##
from dapper import *

##
from collections import namedtuple
import hashlib
from matplotlib import ticker
import logging
mpl_logger = logging.getLogger('matplotlib')


class ExperimentHypercube:
    """View the list of xps as an n-rectangle whose dims correspond to attributes.

    This hyper-rectangle ("hypercube") is of shape: (len(ax) for ax in axes),
    where axes are the distinct attributes of xp_list.

    Normally, this array is highly sparse: array.size >> len(xps),
    coz there are many coordinates with no matching experiment,
    eg. coord(da_method=Climatology, rot=True, ...).

    Therefore, the hypercube IS NOT EXPLICITly represented as an array,
    but kept as a dict whose keys are coordinates.

    Indeed, operations along an axis (optimization or averaging)
    are internally carried out by iterating -- not across the axis --
    but across the the list of xps (and allocating to the appropriate axis tick).

    Since "optimality" is only defined for a given field (eg avrg rmse.a),
    such operations are not done at initialization.
    """

    #----------------------------------
    # Abstract "hypercube" functionality
    #----------------------------------

    def define_coordinate_system(self,axes):
        self.axes = AlignedDict(axes)

        # Accompanying coordinate defintion.
        # Q: Why do we bother with Coord (a namedtuple)
        #    instead of just using a dict of attrs directly?
        # A: To ensure all dict keys conform to the same coordinate system,
        #    which is part of the idea of a "cube".
        Coord = namedtuple('coord', axes)

        # Pretty print
        def str_dict(obj):
            return str(obj).partition("(")[-1].partition(")")[0]
        def str_tuple(obj):
            return ", ".join(str(v) for v in obj._asdict().values())
        Coord.str_dict = str_dict
        Coord.str_tuple = str_tuple

        return Coord

    def __init__(self, axes, _dict=None):
        self.Coord = self.define_coordinate_system(axes)
        self._dict = _dict or dict()

    def as_dict(self):
        return self._dict

    def as_list(self):
        """Relies on python>=3.7 for keeping dict order"""
        return list(self.as_dict().values())

    def __iter__(self):
        for coord in self.as_dict(): yield coord

    def __len__(self):
        return len(self.as_list())

    # TODO: simplify?
    def __contains__(self,key):
        return key in self._dict

    def __setitem__(self,key,val):
        self._dict[key] = val

    def __getitem__(self,key):
        """Indexing."""
        if isinstance(key, dict):
            # Get all items with attrs matching dict
            # Note: implmentation is simpler than for ExperimentList.
            match1 = lambda coord, k: getattr(coord,k)==key[k]
            match  = lambda coord: all(match1(coord,k) for k in key)
            inds   = [i for i,coord in enumerate(self) if match(coord)]
            return self[inds]
        elif isinstance(key,list):
            # Get list of items
            return [self[k] for k in key]
        elif isinstance(key, int) or isinstance(key, slice):
            # Slice
            return self.as_list()[key]
        else:
            # Default: get a single item by its coordinates
            # NB: Shouldn't use isinstance(key, self.Coord)
            # coz it fails when the namedtuple (Coord) has been
            # instantiated in different places (but with equal params).
            # See bugs.python.org/issue7796
            return self._dict[key]

    def get_coord(self,xp):
        """get_coord(), i.e. the inverse of (a clean version of) __getitem__."""
        coord = (getattr(xp,ax,None) for ax in self.axes)
        return self.Coord(*coord)

    def get_tick_inds(self,xp):
        axes = self.axes.items()
        return (ticks.index(getattr(xp,ax)) for ax, ticks in axes)

    # NB: Not tested
    # def prune_ticks(self):
    #     """Eliminate superfluous ticks, but keep ordering."""
    #     for axis, ticks in self.axes.items():
    #         present = [getattr(coord,axis) for coord in self] # default not necessary!
    #         self.axes[axis] = intersect(ticks,*present)

    def add_axis(self, axis):
        if axis in self.axes: return
        d = self._dict # store
        self.__init__( {**self.axes, axis:[None]} )
        self._dict = {self.get_coord(coord):xp for coord, xp in d.items()}

    def nest_spaces(self, inner_axes=None, outer_axes=None):
        """Return a new hypercube with axes `outer_axes`,
        
        obtained by projecting along the ``inner_axes``.
        The entries of this hypercube are themselves hypercubes,
        with axes `inner_axes`,
        each one regrouping the entries with the same (projected) coordinate. 
        """

        # Default: a singleton outer space,
        # with everything contained in the inner (projection) space.
        if inner_axes is None and outer_axes is None:
            outer_axes = ()

        # Validate
        if inner_axes is None:
            inner_axes = complement(self.axes, *outer_axes)
        else:
            assert outer_axes is None
            outer_axes = complement(self.axes, *inner_axes)

        # Include actual axes (ticks)
        outer_axes = {a:self.axes[a] for a in outer_axes}
        inner_axes = {a:self.axes[a] for a in inner_axes}

        # Fill outer cube
        outer_cube = self.__class__(outer_axes)
        for coord in self:
            xp = self[coord]
            outer_coord = outer_cube.get_coord(coord)
            if outer_coord in outer_cube:
                inner_cube = outer_cube[outer_coord]
            else:
                inner_cube = self.__class__(inner_axes)
                outer_cube[outer_coord] = inner_cube
            inner_cube[inner_cube.get_coord(coord)] = xp

        return outer_cube

    def single_out(self,coords,tag=None,NoneAttrs=()):
        """Insert duplicates of self[coords], with a tag.

        This is to distinguish them from all other xps,
        which prevents them being gobbled up in averaging/optimization."""
        xps = []
            
        for xp in self[coords]:
            xp = deepcopy(xp)
            xp.single_out_tag = tag

            # Avoid plotting optimal values.
            for a in NoneAttrs:
                setattr(xp,a,None)

            xps.append(xp)

        # Add axis
        self.add_axis('single_out_tag')
        # Add tag
        assert tag not in self.axes['single_out_tag']
        self.axes['single_out_tag'].append(tag)
        # Add duplicated xps
        self._dict.update( {self.get_coord(xp):xp for xp in xps} )

    def __repr__(self):
        s  = repr_type_and_name(self)
        ID = " "*4 # indent
        a  = repr(self.axes).replace("\n", "\n"+ID)
        a  = a.replace("\n", ID+"\n")
        s  += f" with axes: {a}."
        s  += "\n"+ID + f"The dict has length {len(self)}"
        m  = len(self)//2
        head = [str(x) for x in [*self][:min(m,2)]]
        tail = [str(x) for x in [*self][-min(m,2):]]
        midl = ["..."] if len(head)+len(tail)<len(self) else []
        if len(self):
            s += ("\n"+2*ID).join([" with keys: [", *head, *midl, *tail])
            s += "\n"+ID+"]"
        s += "."
        return s

    #----------------------------------
    # Experiment-specific functionality
    #----------------------------------
    @classmethod
    def from_list(cls, xp_list):

        # Define axes
        xp_list = ExperimentList(xp_list)
        axes = xp_list.split_attrs(nomerge=['single_out_tag'])[0]

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
        axes = make_ticks(axes)

        # Create coordinate system
        obj = cls(axes)
        # Fill "hypercube"
        obj._dict = {obj.get_coord(xp):xp for xp in xp_list}

        return obj

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

        for coord in self:
            xp = self[coord]
            a = get_field(xp)
            avrgs[coord] = a

            found_anything = found_anything or (a is not None)
        if not found_anything: raise RuntimeError(
                f"The stat. field '{statkey}' was not found"
                " among any of the xp's.")
        return avrgs

    def mean_field(self, statkey="rmse.a", axes=None):

        # Note: The case ``axes=()`` should work w/o special treatment.
        if axes is None:
            return self.field(statkey)

        mean_cube = self.nest_spaces(axes)
        for coord in mean_cube:
            group = mean_cube[coord]

            uqs = group.field(statkey)
            vals = [uq.val for uq in uqs.as_list()]

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

            mean_cube[coord] = uq
        return mean_cube

    def tuned_uq(self, axes=None, costfun=None):
        """Get (compile/tabulate) a stat field optimised wrt. tuning params."""

        if axes is None:
            return self

        # Define cost-function
        costfun = (costfun or 'increasing').lower()
        if   'increas' in costfun: costfun = (lambda x: +x)
        elif 'decreas' in costfun: costfun = (lambda x: -x)
        else: assert hasattr(costfun, '__call__') # custom

        tuned_cube = self.nest_spaces(axes)
        for outer_coord in tuned_cube:
            group = tuned_cube[outer_coord]

            # Find optimal value and coord within group
            MIN = np.inf
            for coord in group:
                uq = group[coord]

                cost = costfun(uq.val)
                if cost <= MIN: # inf<=inf is True
                    MIN = cost
                    uq_opt = uq
                    uq_opt.tuning_coord = coord

            tuned_cube[outer_coord] = uq_opt

        return tuned_cube

    def axis_ticks_nn(self,axis_name):
        """Axis ticks without None"""
        return [x for x in self.axes[axis_name] if x is not None]



def load_xps(savepath):
    """Load xp's (as list) from an .xps file or all .xp files in a dir.

    Note: saving this list in a new file (takes considerable time and)
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
