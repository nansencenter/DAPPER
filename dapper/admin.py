"""Define high-level API in DAPPER.

Used for experiment (xp) specification/administration, including:

 - da_method decorator
 - xpList
 - save_data
 - run_experiment
 - run_from_file

 - HiddenMarkovModel
 - Operator
"""

from dapper import *
from textwrap import dedent

import dill
import shutil
from datetime import datetime

class HiddenMarkovModel(NestedPrint):
    """Container for attributes of a Hidden Markov Model (HMM).

    This container contains the specification of a "twin experiment",
    i.e. an "OSSE (observing system simulation experiment)".
    """

    def __init__(self,Dyn,Obs,t,X0,**kwargs):
        self.Dyn = Dyn if isinstance(Dyn, Operator)   else Operator  (**Dyn)
        self.Obs = Obs if isinstance(Obs, Operator)   else Operator  (**Obs)
        self.t   = t   if isinstance(t  , Chronology) else Chronology(**t)
        self.X0  = X0  if isinstance(X0 , RV)         else RV        (**X0)

        # Name
        name = inspect.getfile(inspect.stack()[1][0])
        self.name = str(Path(name).relative_to(rc.dirs.dapper/'mods'))

        # Kwargs
        abbrevs = {'LP':'liveplotters'}
        for key in kwargs:
            setattr(self, abbrevs.get(key,key), kwargs[key])

        # Defaults
        if not hasattr(self.Obs,"localizer"): self.Obs.localizer = no_localization(self.Nx, self.Ny)
        if not hasattr(self    ,"sectors")  : self.sectors       = {}

        # Validation
        if self.Obs.noise.C==0 or self.Obs.noise.C.rk!=self.Obs.noise.C.M:
            raise ValueError("Rank-deficient R not supported.")

    # ndim shortcuts
    @property
    def Nx(self): return self.Dyn.M
    @property
    def Ny(self): return self.Obs.M

    printopts = {'ordering' : ['Dyn','Obs','t','X0']}


    def simulate(self,desc='Truth & Obs'):
        """Generate synthetic truth and observations."""
        Dyn,Obs,chrono,X0 = self.Dyn, self.Obs, self.t, self.X0

        # Init
        xx    = zeros((chrono.K   +1,Dyn.M))
        yy    = zeros((chrono.KObs+1,Obs.M))

        xx[0] = X0.sample(1)

        # Loop
        for k,kObs,t,dt in progbar(chrono.ticker,desc):
            xx[k] = Dyn(xx[k-1],t-dt,dt) + sqrt(dt)*Dyn.noise.sample(1)
            if kObs is not None:
                yy[kObs] = Obs(xx[k],t) + Obs.noise.sample(1)

        return xx,yy



class Operator(NestedPrint):
    """Container for operators (models)."""
    def __init__(self,M,model=None,noise=None,**kwargs):
        self.M = M

        # None => Identity model
        if model is None:
            model = Id_op()
            kwargs['linear'] = lambda x,t,dt: Id_mat(M)
        self.model = model

        # None/0 => No noise
        if isinstance(noise,RV):
            self.noise = noise
        else:
            if noise is None: noise = 0
            if np.isscalar(noise):
                self.noise = GaussRV(C=noise,M=M)
            else:
                self.noise = GaussRV(C=noise)

        # Write attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self,*args,**kwargs):
        return self.model(*args,**kwargs)

    printopts = {'ordering' : ['M','model','noise']}


def da_method(*default_dataclasses):
    """Make the decorator that makes the DA classes.

    Example:
    >>> @dc.dataclass
    >>> class ens_defaults:
    >>>   infl : float = 1.0
    >>>   rot  : bool  = False
    >>> 
    >>> @da_method(ens_defaults)
    >>> class EnKF:
    >>>     N     : int
    >>>     upd_a : str = "Sqrt"
    >>> 
    >>>     def assimilate(self,HMM,xx,yy):
    >>>         ...
    """

    def dataclass_with_defaults(cls):
        """Decorator based on dataclass.

        This adds __init__, __repr__, __eq__, ..., but also includes
        inherited defaults (see stackoverflow.com/a/58130805).

        Also:
         - Wraps assimilate() to provide gentle_fail functionality.
         - Initialises and writes the Stats object.
         - Adds average_stats(), print_averages()."""


        # Default fields invovle: (1) annotations and (2) attributes.
        def set_field(name,type,val):
            if not hasattr(cls,'__annotations__'):
                cls.__annotations__ = {}
            cls.__annotations__[name] = type
            if not isinstance(val,dc.Field):
                val = dc.field(default=val)
            setattr(cls, name, val)

        # APPend default fields without overwriting.
        # Don't implement (by PREpending?) non-default args -- to messy!
        for D in default_dataclasses:
            # NB: Calling dataclass twice always makes repr=True, so avoid this.
            for F in dc.fields(dc.dataclass(D)):
                if F.name not in cls.__annotations__:
                    set_field(F.name,F.type,F)

        # Create new class (NB: old/new classes have same id) 
        orig_assimilate = cls.assimilate # => store old.assimilate
        cls = dc.dataclass(cls)

        # Shortcut for self.__class__.__name__
        cls.da_method = cls.__name__

        # Add instance methods
        def method(fun):
            setattr(cls,fun.__name__,fun)
            return fun

        @method
        @functools.wraps(orig_assimilate)
        def assimilate(self,HMM,xx,yy,desc=None,**stat_kwargs):
            # Progressbar name
            pb_name_hook = self.da_method if desc is None else desc
            # Init stats
            self.stats = Stats(self,HMM,xx,yy,**stat_kwargs)
            # Assimilate
            orig_assimilate(self,HMM,xx,yy)

        @method
        def average_stats(self,free=False):
            """Average (in time) all of the time series in the Stats object.

            If ``free``: del ref to Stats object."""
            self.avrgs = self.stats.average_in_time()
            if free:
                delattr(self,'stats')

        @method
        def print_avrgs(self,keys=()):
            """Tabulated print of averages (as requested by ``keys``)"""
            cfgs = xpList([self])
            cfgs.print_avrgs(keys)

        method(replay)

        return cls
    return dataclass_with_defaults


def run_experiment(xp, label, savedir, HMM,
                   seed=None, free=True, statkeys=False, fail_gently=rc.fail_gently,
                   **stat_kwargs):
    """Used by xpList.launch() to run each single experiment.
    
    This involves steps similar to ``example_1.py``, i.e.:

     - set seed (if not False/None)
     - set any HMM_params
     - hmm.simulate()  : generate truth & obs
     - xp.assimilate() : run DA method, catch exception if fail_gently
     - average_stats() : result averaging
     - print_averages(): result printing
     - dill.dump()     : result storage
    """

    # Set HMM parameters.
    # Why the use of setters? To avoid tying an HMM to each xp, which
    #  - keeps the xp attrs primitive.
    #  - avoids memory and pickling/storage of near-duplicate HMMs.
    HMM_params = intersect(vars(xp), re.compile(r'^HMM_'))
    if HMM_params:
        # We should copy HMM so as not to cause any nasty surprises such as
        # expecting param=1 when param=2 (coz it's not been reset).
        # NB: won't copy implicitly ref'd obj's (like L96's core).
        #     Could yield bug when using MP?
        hmm = deepcopy(HMM) 
        for key in HMM_params:
            val = getattr(xp,key)
            key = key.split('HMM_')[1]
            hmm.param_setters[key](val)
    else:
        hmm = HMM


    # GENERATE TRUTH/OBS
    if seed: set_seed(seed + getattr(xp,'seed',0)) # Set seed
    xx, yy = hmm.simulate()

    # Now set the seed again. Then it does not matter for any following
    # random draws whether-or-not simulate() was called above.
    # Also, offset by a fixed number to prevent any magic correspondence
    # of the random draws of the truth and the state-estimator.
    if seed: set_seed(seed + getattr(xp,'seed',0) + 1)
    # Repeating the seed for each experiment.
    # may yield a form of "Variance reduction"
    # (eg. CRN, see wikipedia.org/wiki/Variance_reduction).
    # Note: this CRN trick is handy for quick comparisons
    #       of results between xp's,
    #       but should not be relied upon for publishing results,
    #       which should simply be converged values.
    

    # ASSIMILATE
    try:
        xp.assimilate(hmm,xx,yy, label, **stat_kwargs)
    except Exception as ERR:
        if fail_gently:
            xp.crashed = True
            print_cropped_traceback(ERR)
        else:
            raise ERR

    # Average in time
    xp.average_stats(free=free)

    # Print
    if statkeys:
        xp.print_avrgs(() if statkeys is True else statkeys)

    # Save
    if savedir:
        with open(savedir/"xp","wb") as FILE:
            dill.dump({'xp':xp}, FILE)


def print_cropped_traceback(ERR):

    def crop_traceback(ERR,lvl):
        msg = []
        try:
            # If IPython, use its coloring functionality
            __IPYTHON__
            from IPython.core.debugger import Pdb
            import traceback as tb
            pdb_instance = Pdb()
            pdb_instance.curframe = inspect.currentframe()

            for i, frame in enumerate(tb.walk_tb(ERR.__traceback__)):
                if i<lvl: continue # skip frames
                if i==lvl: msg += ["   ⋮\n"]
                msg += [pdb_instance.format_stack_entry(frame,context=5)]

        except (NameError,ImportError):
            # No coloring
            for s in traceback.format_tb(ERR.__traceback__):
                msg += "\n".join(s)

        return msg

    msg  = ["\n\nCaught exception during assimilation. Traceback:"]
    msg += ["<"*20 + "\n"]
    msg += crop_traceback(ERR,1) + [str(ERR)]
    msg += ["\n" + ">"*20]
    msg += ["Resuming program execution.\n"
            + "Turn off `fail_gently` to fully raise the exception.\n"]
    for s in msg: print(s,file=sys.stderr)


# TODO: check collections.userlist
# TODO: __add__ vs __iadd__
class xpList(list):
    """List, subclassed for holding experiment ("xp") objects.

    Main use: administrate experiment **launches**.
    Also see: ``xpSpace`` for experiment **result presentation**.

     Modifications to ``list``:

     - ``__iadd__`` (append) also for single items;
       this is hackey, but convenience is king.
     - ``append()`` supports ``unique`` to enable lazy xp declaration.
     - ``__getitem__`` supports lists.
     - pretty printing (using common/distinct attrs).

     Add-ons:

     - ``launch()``
     - ``print_averages()``
     - ``gen_names()``
     - ``inds()`` to search by kw-attrs.
     """

    def __init__(self,*args,unique=False):
        """Initialize without args, or with a list of configs.
         - unique: if true, then duplicates won't get appended."""
        self.unique = unique
        super().__init__(*args)

    def __iadd__(self,cfg):
        if not hasattr(cfg,'__iter__'):
            cfg = [cfg]
        for item in cfg:
            self.append(item)
        return self

    def append(self,cfg):
        "Append if not unique&present"
        if not (self.unique and cfg in self): super().append(cfg)

    def __getitem__(self, keys):
        """Indexing, also by a list"""
        try:              B=[self[k] for k in keys]   # if keys is list
        except TypeError: B=super().__getitem__(keys) # if keys is int, slice
        if hasattr(B,'__len__'): B = xpList(B) # Cast
        return B 

    def inds(self,strict=True,missingval="NONSENSE",**kws):
        """Find (all) indices of configs whose attributes match kws.

        If strict, then xp's lacking a requested attr will not match,
        unless the missingval (e.g. None) matches the required value.
        """
        def match(xp):
            missing = lambda v: missingval if strict else v
            matches = [getattr(xp,k,missing(v))==v for k,v in kws.items()]
            return all(matches)

        return [i for i,xp in enumerate(self) if match(xp)]

    @property
    def da_methods(self):
        return [xp.da_method for xp in self]

    def split_attrs(self,nomerge=()):
        """Compile the attrs of all xps; split as distinct, redundant, common.

        Insert None if an attribute is distinct but not in xp."""

        def _aggregate_keys():
            "Aggregate keys from all xps"

            if len(self)==0: return []

            # Start with da_method
            aggregate = ['da_method']

            # Aggregate all other keys
            for xp in self:

                # Get dataclass fields
                try:
                    dc_fields = dc.fields(xp.__class__)
                    dc_names = [F.name for F in dc_fields]
                    keys = xp.__dict__.keys()
                except TypeError:
                    # Assume namedtuple
                    dc_names = []
                    keys = xp._fields

                # For all potential keys:
                for k in keys:
                    # If not already present:
                    if k not in aggregate:

                        # If dataclass, check repr:
                        if k in dc_names:
                            if dc_fields[dc_names.index(k)].repr:
                                aggregate.append(k)
                        # Else, just append
                        else:
                            aggregate.append(k)

            # Remove unwanted
            excluded  = [re.compile('^_'),'avrgs','stats','HMM']
            aggregate = complement(aggregate,*excluded)
            return aggregate

        distinct, redundant, common = {}, {}, {}

        for key in _aggregate_keys():

            # Want to distinguish actual None's from empty ("N/A").
            # => Don't use getattr(obj,key,None)
            vals = [getattr(xp,key,"N/A") for xp in self]

            # Sort (assign dct) into distinct, redundant, common
            if flexcomp(key, *nomerge):
                # nomerge => Distinct
                dct, vals = distinct, vals
            elif all(vals[0]==v for v in vals):
                # all values equal => common
                dct, vals = common, vals[0]
            else:
                v0 = next(v for v in vals if "N/A"!=v)
                if all(v=="N/A" or v==v0 for v in vals):
                    # all values equal or "N/A" => redundant
                    dct, vals = redundant, v0
                else:
                    # otherwise => distinct
                    dct, vals = distinct, vals

            # Replace "N/A" by None
            sub = lambda v: None if v=="N/A" else v
            if isinstance(vals,str): vals = sub(vals)
            else:
                try:                 vals = [sub(v) for v in vals]
                except TypeError:    vals = sub(vals)

            dct[key] = vals

        return distinct, redundant, common

    def __repr__(self):
        distinct, redundant, common = self.split_attrs()
        s = '<xpList> of length %d with attributes:\n'%len(self)
        s += tabulate(distinct)
        s += "\nOther attributes:\n"
        s += str(AlignedDict({**redundant, **common}))
        return s

    @functools_wraps(tabulate_avrgs)
    def _repr_avrgs(self,*args,**kwargs): 
        """Pretty (tabulated) repr of cfgs & avrgs (val±conf)."""
        distinct, redundant, common = self.split_attrs()

        # Prepare table components
        headr1, mattr1 = list(distinct.keys()), list(distinct.values())
        headr2, mattr2 = tabulate_avrgs([C.avrgs for C in self],*args,**kwargs,pad='æ')
        # Join 1&2
        headr = headr1 + ['|']             + headr2
        mattr = mattr1 + [['|']*len(self)] + mattr2

        table = tabulate(mattr, headr).replace('æ',' ')
        return table

    # TODO: rm _repr_avrgs (implement it here)
    # and make out of stdout capturing in the tests.
    @functools.wraps(_repr_avrgs)
    def print_avrgs(self,*args,**kwargs):
        print(self._repr_avrgs(*args,**kwargs))

    def gen_names(self,abbrev=4,tab=False):
        """Similiar to ``self.__repr__()``, but:

          - returns *list* of names
          - attaches label to each attribute
          - tabulation is only an option
          - abbreviates labels to width abbrev
        """
        distinct, redundant, common = self.split_attrs(nomerge=["da_method"])
        labels = distinct.keys()
        values = distinct.values()

        # Label abbreviation
        labels = [collapse_str(k,abbrev) for k in labels]

        # Make label columns: insert None or lbl+":", depending on value
        column = lambda  lbl,vals: [None if v is None else lbl+":" for v in vals]
        labels = [column(lbl,vals) for lbl, vals in zip(labels,values)]

        # Interlace labels and values
        table = [x for (a,b) in zip(labels,values) for x in (a,b)]

        # Rm da_method label (but keep value)
        table.pop(0)

        # Tabulate
        table = tabulate(table,inds=False, tablefmt="plain")

        # Rm space between lbls/vals
        table = re.sub(':  +',':',table) 

        # Rm alignment
        if not tab:
            table = re.sub(r' +',r' ', table)

        return table.splitlines()

    def launch(self, HMM, seed="clock", save_as="noname", mp=False, **kwargs):
        """For each xp in self: run_experiment(xp, ...).
        
        Value of `seed` ||| Consequent seed mngmt. by run_experiment():
        ----------------|||---------------------------------------------
        False/None      ||| No management.
        a number>0      ||| Uses xp.seed + seed.
        True/"clock"    ||| Uses xp.seed + set_seed("clock")

        The results are saved in ``rc.dirs['data']/save_as.stem``,
        unless ``save_as`` is False/None.

        Depending on ``mp``, run_experiment() is delegated to one of:
         - caller process (no parallelisation)
         - multiprocessing on this host
         - GCP (Google Cloud Computing) with HTCondor

        The kwargs are forwarded to run_experiment().
        
        See ``example_2.py`` and ``example_3.py`` for example use.
        """

        # Include in kwargs
        kwargs['HMM'] = HMM
        kwargs['seed'] = set_seed("clock") if seed in ["clock",True] else seed

        if save_as in [None,False]:
            assert not mp, "Multiprocessing requires saving data."
            # Parallelization w/o storing is possible, especially w/ threads.
            # But it involves more complicated communication setup.
            xpi_dir = lambda *args: None
        else:
            save_as = rc.dirs.data / Path(save_as).stem
            save_as /= "run_" + datetime.now().strftime("%Y-%m-%d__%H:%M:%S")
            os.makedirs(save_as)
            print(f"Experiment data stored at {save_as}")
            def xpi_dir(i):
                path = save_as / str(i)
                os.mkdir(path)
                return path


        # No parallelization
        if not mp: 
            for ixp, (xp, label) in enumerate( zip(self, self.gen_names()) ):
                run_experiment(xp, label, xpi_dir(ixp), **kwargs)


        # Local multiprocessing
        elif mp in ["MP", True]:

            def run_with_fixed_args(arg):
                xp, ixp = arg
                run_experiment(xp, None, xpi_dir(ixp), **kwargs)
            args = zip(self, range(len(self)))

            with     set_tmp(tools.utils,'disable_progbar',True):
                with set_tmp(tools.utils,'disable_user_interaction',True):
                    NPROC = None # None => multiprocessing.cpu_count()
                    with mpd.Pool(NPROC) as pool:
                        list(tqdm.tqdm(pool.imap(run_with_fixed_args, args),
                            total=len(self), desc="Parallel experim's", smoothing=0.1))


        # GCP
        elif isinstance(mp,dict):

            for ixp, xp in enumerate(self):
                with open(xpi_dir(ixp)/"xp.var","wb") as f:
                    dill.dump(dict(xp=xp), f)

            with open(save_as/"xp.com","wb") as f:
                dill.dump(kwargs, f)

            extra_files = save_as / "extra_files"
            os.mkdir(extra_files)
            for f in mp["extra_files"]:
                if isinstance(f, (str,PurePath)):
                    # Example: f = "A.py"
                    path = Path(sys.path[0]) / f
                    dst = f
                else:
                    # Example: f = ("~/E/G/A.py", "G")
                    path, root = f
                    dst = Path(path).relative_to(root)
                dst = extra_files / dst
                os.makedirs(dst.parent, exist_ok=True)
                try:            shutil.copytree(path, dst) # dir -r
                except OSError: shutil.copy2   (path, dst) # file

            # com["fail_gently"] = True # TODO: make this unnecesary

            # Loads PWD/xp_{var,com} and calls run_experiment()
            with open(extra_files/"load_and_run.py","w") as f:
                f.write( dedent("""\
                from dapper import *

                # Load
                with open("xp.com", "rb") as f: com = dill.load(f)
                with open("xp.var", "rb") as f: var = dill.load(f)

                # User-defined code
                %s

                # Run
                result = run_experiment(var['xp'], None, Path("."), **com)
                """)%dedent(mp["code"]) )

            submit_job_GCP(save_as)



        return save_as
