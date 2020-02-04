"""Test data loading and presenting functionality."""

##

# TODO:
# - Subclass dict for xpSpace
# - Make core part of xpSpace into HyperSpace, and subclass it
# - Write uqs to avrg field?
# - Should take mean before splitting into cols?
# - Replace header argument in tabulate_column with minWidth.
# - Fix ax vs axis vs axes
# - Rename xp_list to as_list
# - Rename xp_dict to as_dict
# - rename xpSpace, xpList? Also remember "hypercube", "cube", etc...
# - Update plot_1d
# - replace all np.vectorize in math.py with vectorize0

# In data_management.py:
# - In nest_spaces():
#    - rename elsewhere, grep NULL
#    - docstring update
# - In tuned_field():
#   AFAICT, mean_coord gets re-produced and re-checked
#   for all seeds, which is unnecessary.

##
from dapper import *
##

# Enable storing None's (not as nan's) both in
# - structured np.array
# - pd.DataFrame
_otype = object # could also use 'O'

def unpack_uqs(lst, decimals=None):

    subcols=["val","conf","nFail","nSuccess","tuning_coord"]

    # np.array with named columns.
    dtype = np.dtype([(c,_otype) for c in subcols])
    avrgs = np.full_like(lst, dtype=dtype, fill_value=None)

    for i,uq in enumerate(lst):
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

_DEFAULT_ALLOTMENT = dict(
        outer=None,
        inner=None,
        mean=None,
        optim=None,
        )

def nest_xps(hypercube, statkey="rmse.a", axes=_DEFAULT_ALLOTMENT):
    # TODO: (imporove this comment)
    # Note: Observe the sequence of the above
    # (data>tables>mean>optim>cols).
    # Each level makes a call to nest_spaces() [mean_field, opt_field also].
    # The rest of this function is formatting,
    # and could be swapped for plotting functionality.

    # Note: cannot support multiple statkeys
    #       because it's not (obviously) meaningful
    #       when optimizing over tuning_axes.

    # Validate axes
    roles = {} # "inv"
    for role in set(axes) | set(_DEFAULT_ALLOTMENT):
        assert role in _DEFAULT_ALLOTMENT, f"Invalid role {role!r}"
        aa = axes.get(role,_DEFAULT_ALLOTMENT[role])

        if aa is None:
            pass # Purposely special
        else:
            # Ensure iterable
            if isinstance(aa,str) or not hasattr(aa,"__iter__"):
                aa = (aa,)

            for axis in aa:
                # Ensure valid axis name
                assert axis in hypercube.axes, f"Axis {axis!r} not among hypercube.axes."
                # Ensure unique
                if axis in roles:
                    raise TypeError(f"An axis (here {axis!r}) cannot be assigned"
                    f" to 2 roles (here {role!r} and {roles[axis]!r}).")
                else:
                    roles[axis] = role
        axes[role] = aa

    # Split into tables
    tables = hypercube.nest_spaces(outer_axes=axes['outer'])
    for table_coord in tables:
        table = tables[table_coord]
        # Average (or don't)
        table = table.mean_field(statkey, axes['mean'])
        # Optimize (or don't)
        table_tuned = table.tuned_uq(axes['optim'])
        # Split into columns
        columns = table_tuned.nest_spaces(outer_axes=axes['inner'])

        # Overwrite table
        # by its decomposition into (tuned, avrg'd) columns
        tables[table_coord] = columns

        # Make row_keys (i.e. left part of the table)
        # by computing `distinct` after eliminating inner axis.
        rows = table_tuned.nest_spaces(inner_axes=axes['inner'] or ())
        columns.rows = rows # TODO: rm
        rows = ExperimentList([*rows])
        distinct = rows.split_attrs()[0]
        columns.row_keys = distinct

    return axes, tables

import pandas as pd

def print_1d(hypercube, statkey="rmse.a", axes=_DEFAULT_ALLOTMENT, subcols=True):
    """Print table of results.

    - statkey: The statistical field from the experiments to report.

    - subcols: If True, then subcolumns are added to indicate the
               1σ confidence interval, and potentially some other stuff.

    - axes: Allots (maps) each role to a set of axis of the hypercube.
      Suggestion: dict(outer='da_method', inner='N', mean='seed', optim=('infl','loc_rad'))

      Example: If ``mean`` is assigned to:

      - ("seed",): Experiments are averaged accross seeds, and the 1σ (sub)col is
                   computed as sqrt(var(xps)/N) where xps is a set of experiments.

      - ()       : Experiments are averaged across nothing
                   (i.e.) this is a valid edge case of the previous one.

      - None     : Experiments are not averaged, and the 1σ (sub)col reports the
                   confidence from that individual experiment (time series of values).
    """

    axes, tables = nest_xps(hypercube,statkey,axes)

    # Used many times ==> abbreviate
    mn = axes['mean']  is not None
    tu = axes['optim'] is not None

    for table_coord in tables:
        table = tables[table_coord]

        # Format row_keys
        row_keys = table.row_keys
        row_keys = pd.DataFrame.from_dict(row_keys,dtype=_otype)
        if row_keys.empty:
            # If there's only 1 xp, row_keys/distinct will be {},
            # but really it should have been the following,
            # which has length one:
            row_keys = pd.DataFrame(index=[0])
        header_is2 = bool(axes['inner'])
        SEPa = [ '|' + ('\n|' if header_is2 else '')]
        SEPb = [['|']*len(row_keys)]
        headers = [('\n' if header_is2 else '')+k for k in row_keys] + SEPa
        matters = [list(row_keys[k])              for k in row_keys] + SEPb
        is_1st_col = True

        # Loop over table (as columns)
        for j, col_coord in enumerate(table):
            column = table[col_coord]

            # Extract list of uqs from column.
            uqs = []
            for i, row in row_keys.iterrows():
                uqs += column[row.to_dict()] or [None]
            assert len(uqs)>0 # good 4 debugging
            # Convert to structured array
            column = unpack_uqs(uqs)

            # Tabulate (sub)columns
            if subcols:

                subc = dict()
                subc['keys']     = ["val"   , "conf"]
                subc['headers']  = [statkey , '1σ']
                subc['frmts']    = [None    , None]
                subc['spaces']   = [' ±'    , ] # last one gets appended below.
                subc['aligns']   = ['>'     , '<'] # 4 header -- matter gets decimal-aligned.
                if tu:
                    subc['keys']    += ["tuning_coord"]
                    subc['headers'] += [axes['optim']]
                    subc['frmts']   += [lambda x: tuple(a for a in x)]
                    subc['spaces']  += [' *']
                    subc['aligns']  += ['<']
                elif mn:
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

            # Super-header
            if axes['inner']:
                if  is_1st_col:
                    is_1st_col = False
                    col_header = col_coord.str_dict()
                else:
                    col_header = col_coord.str_tuple()
                super_header = col_header
                width = len(header)
                # if subcols and mn: width += 1 # coz ✔️ takes 2 chars
                super_header = super_header.center(width,"_")
                header = super_header + "\n" + header

            # Append column to table
            matters = matters + [matter]
            headers = headers + [header]

        table = tabulate(matters, headers).replace('æ',' ')

        # Print
        if axes['outer']:
            table_title = "•Table for " + table_coord.str_dict() + "."
            table_title = table_title + (f" •Averages Σ over {axes['mean']}." if axes['mean'] else "")
            with coloring(termcolors['underline']):
                print("\n" + table_title)
        print(table)

##
import inspect
import io
from contextlib import redirect_stdout
import functools

# Test functions will be registering here.
lcls = locals()
test_ind = 0


if "--replace" in sys.argv:
    replacements = []

    import dataclasses as dc
    @dc.dataclass
    class Replacement:
        lines  : list
        nOpen  : int
        nClose : int

    with open(__file__,"r") as F:
        orig_code = [ln for ln in F]

    def backtrack_until_finding(substr,lineno):
        while True:
            lineno -= 1
            if substr in orig_code[lineno]:
                return lineno


##
@functools.wraps(print_1d)
def _print_1d(*args,**kwargs):
    """Usage: As the usual print_1d, but makes it into a test.

    Options:
        --replace : update this file (i.e. its data) 
        --print   : turn _print_1d into print_1d.

    Features:
    - Enables re-use of ``old`` variable name (by capturing its value).
    - Parameterization -- pytest.mark.parametrize not used.
                          Avoids having to decorate an explicit function
                          (and thus enables naming functions through test_ind).
    - Capturing stdout -- The func print_1d() is only called once for each ``old``
                          (unlike a pytest fixture with capsys),
                          and thus it's fast.
    - splitlines() included.

    Obsolete?:
        the strip() functionality is used to remove bothersome trailing whitespaces
        (usually not printed by terminal, and so not copied when doing it manually)
    """

    if "--print" in sys.argv:
        print_1d(*args,**kwargs)
        return

    # Call print_1d(). Capture stdout.
    F = io.StringIO()
    with redirect_stdout(F):
        print_1d(*args,**kwargs)
    printed_lines = F.getvalue().splitlines(True)

    if "--replace" in sys.argv:
        caller_lineno = inspect.currentframe().f_back.f_lineno
        nClose = backtrack_until_finding('"""\n', caller_lineno)
        nOpen  = backtrack_until_finding('"""', nClose)
        replacements.append(Replacement(printed_lines,nOpen,nClose))

    else: # Generate & register tests
        global test_ind
        test_ind += 1

        # Capture ``old``
        _old = old.splitlines(True) # keepends

        # Loop over rows
        for lineno, (old_bound,new_bound) in enumerate(zip(_old,printed_lines)):

            # Define test function.
            def compare(old_line=old_bound,new_line=new_bound):
                assert old_line == new_line
                # assert old_line.strip() == new_line.strip()

            # Register test
            lcls[f'test_{test_ind}_line_{lineno}'] = compare


##
# Mac:
# savepath = "~/Desktop/dpr_data/example_3/run_2020-01-27_10-46-59"
# savepath = '~/Desktop/dpr_data/example_3/run_2020-01-27_11-46-33'
#
# P2721L:
# savepath = "/home/pnr/dpr_data/example_3/run_2020-01-03_18-51-09"
__file__ = "tests/test_data.py"
savepath = save_dir(__file__)
xps = load_xps(savepath)
xps = ExperimentHypercube.from_list(xps)

xps_shorter = ExperimentHypercube.from_list([xp for xp in xps.as_list()
    if getattr(xp,'da_method')!='LETKF'])

##
old = """[4m
•Table for da_method='Climatology'. •Averages Σ over ('seed',).[0m
     |  ______N=None_____
     |  rmse.a ±1σ    ☠ ✓
---  -  -----------------
[0]  |   3.624 ±0.006 0 3
[4m
•Table for da_method='OptInterp'. •Averages Σ over ('seed',).[0m
     |  ______N=None_____
     |  rmse.a ±1σ    ☠ ✓
---  -  -----------------
[0]  |   0.941 ±0.001 0 3
[4m
•Table for da_method='EnKF'. •Averages Σ over ('seed',).[0m
           |  ______N=10______  _______12_______  _______14_______
     infl  |  rmse.a ±1σ   ☠ ✓  rmse.a ±1σ   ☠ ✓  rmse.a ±1σ   ☠ ✓
---  ----  -  ----------------  ----------------  ----------------
[0]  1     |    4.55 ±0.09 0 3   4.33  ±0.07 0 3    4.19 ±0.08 0 3
[1]  1.01  |    4.4  ±0.04 0 3   4.24  ±0.1  0 3    3.96 ±0.1  0 3
[2]  1.02  |    4.36 ±0.06 0 3   4.08  ±0.2  0 3    3.76 ±0.1  0 3
[3]  1.04  |    4.16 ±0.1  0 3   3.88  ±0.07 0 3    2.88 ±0.3  0 3
[4]  1.07  |    3.92 ±0.2  0 3   3.6   ±0.2  0 3    2.7  ±0.3  0 3
[5]  1.1   |    3.84 ±0.1  0 3   3.462 ±0.03 0 3    2.28 ±0.3  0 3
[6]  1.2   |    3.58 ±0.1  0 3   2.92  ±0.05 0 3    1.28 ±0.2  0 3
[7]  1.4   |    3.43 ±0.06 0 3   2.52  ±0.1  0 3    0.92 ±0.2  0 3
[4m
•Table for da_method='LETKF'. •Averages Σ over ('seed',).[0m
                     |  _______N=10______  ________12_______  ________14_______
      infl  loc_rad  |  rmse.a ±1σ    ☠ ✓  rmse.a ±1σ    ☠ ✓  rmse.a ±1σ    ☠ ✓
----  ----  -------  -  -----------------  -----------------  -----------------
[0]   1         0.1  |  3.758  ±0.01  0 3  3.756  ±0.01  0 3  3.755  ±0.009 0 3
[1]   1         0.4  |  0.51   ±0.03  0 3  0.53   ±0.07  0 3  0.53   ±0.08  0 3
[2]   1         2    |  1.3    ±0.6   0 3  0.8    ±0.5   0 3  0.9    ±0.3   0 3
[3]   1.01      0.1  |  3.818  ±0.01  0 3  3.806  ±0.01  0 3  3.8    ±0.01  0 3
[4]   1.01      0.4  |  0.41   ±0.01  0 3  0.408  ±0.02  0 3  0.396  ±0.01  0 3
[5]   1.01      2    |  0.26   ±0.007 0 3  0.251  ±0.008 0 3  0.25   ±0.009 0 3
[6]   1.02      0.1  |  3.892  ±0.01  0 3  3.868  ±0.01  0 3  3.86   ±0.01  0 3
[7]   1.02      0.4  |  0.3752 ±0.004 0 3  0.375  ±0.005 0 3  0.3696 ±0.004 0 3
[8]   1.02      2    |  0.245  ±0.007 0 3  0.242  ±0.006 0 3  0.24   ±0.004 0 3
[9]   1.04      0.1  |  4.118  ±0.01  0 3  4.08   ±0.006 0 3  4.064  ±0.02  0 3
[10]  1.04      0.4  |  0.3616 ±0.004 0 3  0.36   ±0.003 0 3  0.3594 ±0.003 0 3
[11]  1.04      2    |  0.2454 ±0.003 0 3  0.2454 ±0.003 0 3  0.2448 ±0.003 0 3
[12]  1.07      0.1  |  nan    ±nan   3 0  nan    ±nan   3 0  nan    ±nan   3 0
[13]  1.07      0.4  |  0.3726 ±0.003 0 3  0.3732 ±0.003 0 3  0.3702 ±0.003 0 3
[14]  1.07      2    |  0.2664 ±0.002 0 3  0.2664 ±0.002 0 3  0.2664 ±0.002 0 3
[15]  1.1       0.1  |  nan    ±nan   3 0  nan    ±nan   3 0  nan    ±nan   3 0
[16]  1.1       0.4  |  0.3932 ±0.002 0 3  0.393  ±0.003 0 3  0.3924 ±0.002 0 3
[17]  1.1       2    |  0.2904 ±0.001 0 3  0.2916 ±0.001 0 3  0.2928 ±0.002 0 3
[18]  1.2       0.1  |  nan    ±nan   3 0  nan    ±nan   3 0  nan    ±nan   3 0
[19]  1.2       0.4  |  0.468  ±0.002 0 3  0.4692 ±0.002 0 3  0.4694 ±0.001 0 3
[20]  1.2       2    |  0.3716 ±0.002 0 3  0.3744 ±0.001 0 3  0.3762 ±0.001 0 3
[21]  1.4       0.1  |  nan    ±nan   3 0  nan    ±nan   3 0  nan    ±nan   3 0
[22]  1.4       0.4  |  0.5872 ±0.001 0 3  0.5884 ±0.002 0 3  0.5892 ±0.001 0 3
[23]  1.4       2    |  0.4978 ±0.001 0 3  0.5022 ±0.001 0 3  0.5052 ±0.001 0 3
"""
_print_1d(xps, "rmse.a", dict(outer="da_method",inner="N",mean="seed",))

##
old = """[4m
•Table for da_method='Climatology'. •Averages Σ over ('seed',).[0m
     |  _________N=None_________
     |  rmse.a ±1σ    *('infl',)
---  -  ------------------------
[0]  |   3.624 ±0.006 *(None,)  
[4m
•Table for da_method='OptInterp'. •Averages Σ over ('seed',).[0m
     |  _________N=None_________
     |  rmse.a ±1σ    *('infl',)
---  -  ------------------------
[0]  |   0.941 ±0.001 *(None,)  
[4m
•Table for da_method='EnKF'. •Averages Σ over ('seed',).[0m
     |  __________N=10_________  __________12__________  __________14__________
     |  rmse.a ±1σ   *('infl',)  rmse.a ±1σ  *('infl',)  rmse.a ±1σ  *('infl',)
---  -  -----------------------  ----------------------  ----------------------
[0]  |    3.43 ±0.06 *(1.4,)       2.52 ±0.1 *(1.4,)       0.92 ±0.2 *(1.4,)   
[4m
•Table for da_method='LETKF'. •Averages Σ over ('seed',).[0m
              |  __________N=10__________  ___________12___________  ___________14___________
     loc_rad  |  rmse.a ±1σ    *('infl',)  rmse.a ±1σ    *('infl',)  rmse.a ±1σ    *('infl',)
---  -------  -  ------------------------  ------------------------  ------------------------
[0]      0.1  |  3.758  ±0.01  *(1.0,)      3.756 ±0.01  *(1.0,)     3.755  ±0.009 *(1.0,)   
[1]      0.4  |  0.3616 ±0.004 *(1.04,)     0.36  ±0.003 *(1.04,)    0.3594 ±0.003 *(1.04,)  
[2]      2    |  0.245  ±0.007 *(1.02,)     0.242 ±0.006 *(1.02,)    0.24   ±0.004 *(1.02,)  
"""
_print_1d(xps, "rmse.a", dict(outer="da_method",inner="N",mean="seed",optim="infl"))

##
old = """[4m
•Table for da_method='Climatology'. •Averages Σ over ('seed',).[0m
     |  _____N=None____
     |  kurt.f ±1σ  ☠ ✓
---  -  ---------------
[0]  |  nan    ±nan 3 0
[4m
•Table for da_method='OptInterp'. •Averages Σ over ('seed',).[0m
     |  _____N=None____
     |  kurt.f ±1σ  ☠ ✓
---  -  ---------------
[0]  |  nan    ±nan 3 0
[4m
•Table for da_method='EnKF'. •Averages Σ over ('seed',).[0m
           |  ________N=10_______  _________12________  _________14________
     infl  |   kurt.f ±1σ     ☠ ✓   kurt.f ±1σ     ☠ ✓   kurt.f ±1σ     ☠ ✓
---  ----  -  -------------------  -------------------  -------------------
[0]  1     |  -1.0138 ±0.001  0 3  -0.8604 ±0.003  0 3  -0.7544 ±0.002  0 3
[1]  1.01  |  -1.01   ±0.0007 0 3  -0.8658 ±0.001  0 3  -0.76   ±0.0006 0 3
[2]  1.02  |  -1.0128 ±0.002  0 3  -0.862  ±0.002  0 3  -0.758  ±0.002  0 3
[3]  1.04  |  -1.0122 ±0.001  0 3  -0.8669 ±0.0008 0 3  -0.758  ±0.001  0 3
[4]  1.07  |  -1.01   ±0.002  0 3  -0.8658 ±0.001  0 3  -0.7576 ±0.002  0 3
[5]  1.1   |  -1.013  ±0.0008 0 3  -0.8646 ±0.003  0 3  -0.7576 ±0.002  0 3
[6]  1.2   |  -1.008  ±0.003  0 3  -0.8704 ±0.004  0 3  -0.7576 ±0.004  0 3
[7]  1.4   |  -1.0104 ±0.004  0 3  -0.8628 ±0.003  0 3  -0.753  ±0.001  0 3
[4m
•Table for da_method='LETKF'. •Averages Σ over ('seed',).[0m
                     |  ________N=10________  _________12_________  _________14_________
      infl  loc_rad  |    kurt.f ±1σ     ☠ ✓    kurt.f ±1σ     ☠ ✓    kurt.f ±1σ     ☠ ✓
----  ----  -------  -  --------------------  --------------------  --------------------
[0]   1         0.1  |  -1.0162  ±0.001  0 3  -0.874   ±0.0007 0 3  -0.7679  ±0.0009 0 3
[1]   1         0.4  |  -1.008   ±0.001  0 3  -0.86    ±0.006  0 3  -0.75    ±0.003  0 3
[2]   1         2    |  -1.0128  ±0.002  0 3  -0.8664  ±0.003  0 3  -0.7552  ±0.004  0 3
[3]   1.01      0.1  |  -1.0158  ±0.001  0 3  -0.8684  ±0.001  0 3  -0.76266 ±0.0003 0 3
[4]   1.01      0.4  |  -1.0094  ±0.001  0 3  -0.8584  ±0.002  0 3  -0.7555  ±0.0006 0 3
[5]   1.01      2    |  -1.011   ±0.003  0 3  -0.8632  ±0.004  0 3  -0.7608  ±0.003  0 3
[6]   1.02      0.1  |  -1.0108  ±0.002  0 3  -0.8676  ±0.0003 0 3  -0.762   ±0.003  0 3
[7]   1.02      0.4  |  -1.0104  ±0.0005 0 3  -0.866   ±0.0004 0 3  -0.7542  ±0.003  0 3
[8]   1.02      2    |  -1.011   ±0.003  0 3  -0.864   ±0.003  0 3  -0.758   ±0.002  0 3
[9]   1.04      0.1  |  -1.012   ±0.001  0 3  -0.86472 ±0.0004 0 3  -0.7564  ±0.0002 0 3
[10]  1.04      0.4  |  -1.0084  ±0.002  0 3  -0.8592  ±0.003  0 3  -0.7518  ±0.003  0 3
[11]  1.04      2    |  -1.0086  ±0.001  0 3  -0.864   ±0.003  0 3  -0.7604  ±0.002  0 3
[12]  1.07      0.1  |  nan      ±nan    3 0  nan      ±nan    3 0  nan      ±nan    3 0
[13]  1.07      0.4  |  -1.0084  ±0.002  0 3  -0.8636  ±0.002  0 3  -0.7532  ±0.002  0 3
[14]  1.07      2    |  -1.0083  ±0.0007 0 3  -0.8699  ±0.0009 0 3  -0.7588  ±0.001  0 3
[15]  1.1       0.1  |  nan      ±nan    3 0  nan      ±nan    3 0  nan      ±nan    3 0
[16]  1.1       0.4  |  -1.0098  ±0.003  0 3  -0.8632  ±0.001  0 3  -0.7548  ±0.003  0 3
[17]  1.1       2    |  -1.0105  ±0.0006 0 3  -0.8652  ±0.001  0 3  -0.7576  ±0.002  0 3
[18]  1.2       0.1  |  nan      ±nan    3 0  nan      ±nan    3 0  nan      ±nan    3 0
[19]  1.2       0.4  |  -1.008   ±0.001  0 3  -0.8632  ±0.0006 0 3  -0.7536  ±0.001  0 3
[20]  1.2       2    |  -1.0109  ±0.0009 0 3  -0.8667  ±0.0007 0 3  -0.75864 ±0.0004 0 3
[21]  1.4       0.1  |  nan      ±nan    3 0  nan      ±nan    3 0  nan      ±nan    3 0
[22]  1.4       0.4  |  -1.00992 ±0.0004 0 3  -0.8632  ±0.002  0 3  -0.7566  ±0.003  0 3
[23]  1.4       2    |  -1.011   ±0.001  0 3  -0.868   ±0.002  0 3  -0.7596  ±0.001  0 3
"""
_print_1d(xps, "kurt.f", dict(outer="da_method",inner="N",mean="seed",))

##
old = """[4m
•Table for da_method='Climatology'. •Averages Σ over ('seed',).[0m
     |  ____infl=None____
     |  rmse.a ±1σ    ☠ ✓
---  -  -----------------
[0]  |   3.624 ±0.006 0 3
[4m
•Table for da_method='OptInterp'. •Averages Σ over ('seed',).[0m
     |  ____infl=None____
     |  rmse.a ±1σ    ☠ ✓
---  -  -----------------
[0]  |   0.941 ±0.001 0 3
[4m
•Table for da_method='EnKF'. •Averages Σ over ('seed',).[0m
         |  ____infl=1.0____  ______1.01______  ______1.02______  ______1.04______  ______1.07_____  ______1.1_______  ______1.2_______  ______1.4_______
      N  |  rmse.a ±1σ   ☠ ✓  rmse.a ±1σ   ☠ ✓  rmse.a ±1σ   ☠ ✓  rmse.a ±1σ   ☠ ✓  rmse.a ±1σ  ☠ ✓  rmse.a ±1σ   ☠ ✓  rmse.a ±1σ   ☠ ✓  rmse.a ±1σ   ☠ ✓
---  --  -  ----------------  ----------------  ----------------  ----------------  ---------------  ----------------  ----------------  ----------------
[0]  10  |    4.55 ±0.09 0 3    4.4  ±0.04 0 3    4.36 ±0.06 0 3    4.16 ±0.1  0 3    3.92 ±0.2 0 3   3.84  ±0.1  0 3    3.58 ±0.1  0 3    3.43 ±0.06 0 3
[1]  12  |    4.33 ±0.07 0 3    4.24 ±0.1  0 3    4.08 ±0.2  0 3    3.88 ±0.07 0 3    3.6  ±0.2 0 3   3.462 ±0.03 0 3    2.92 ±0.05 0 3    2.52 ±0.1  0 3
[2]  14  |    4.19 ±0.08 0 3    3.96 ±0.1  0 3    3.76 ±0.1  0 3    2.88 ±0.3  0 3    2.7  ±0.3 0 3   2.28  ±0.3  0 3    1.28 ±0.2  0 3    0.92 ±0.2  0 3
[4m
•Table for da_method='LETKF'. •Averages Σ over ('seed',).[0m
                  |  _____infl=1.0____  _______1.01______  _______1.02______  _______1.04______  _______1.07______  _______1.1_______  _______1.2_______  _______1.4_______
      N  loc_rad  |  rmse.a ±1σ    ☠ ✓  rmse.a ±1σ    ☠ ✓  rmse.a ±1σ    ☠ ✓  rmse.a ±1σ    ☠ ✓  rmse.a ±1σ    ☠ ✓  rmse.a ±1σ    ☠ ✓  rmse.a ±1σ    ☠ ✓  rmse.a ±1σ    ☠ ✓
---  --  -------  -  -----------------  -----------------  -----------------  -----------------  -----------------  -----------------  -----------------  -----------------
[0]  10      0.1  |   3.758 ±0.01  0 3   3.818 ±0.01  0 3  3.892  ±0.01  0 3  4.118  ±0.01  0 3  nan    ±nan   3 0  nan    ±nan   3 0  nan    ±nan   3 0  nan    ±nan   3 0
[1]  10      0.4  |   0.51  ±0.03  0 3   0.41  ±0.01  0 3  0.3752 ±0.004 0 3  0.3616 ±0.004 0 3  0.3726 ±0.003 0 3  0.3932 ±0.002 0 3  0.468  ±0.002 0 3  0.5872 ±0.001 0 3
[2]  10      2    |   1.3   ±0.6   0 3   0.26  ±0.007 0 3  0.245  ±0.007 0 3  0.2454 ±0.003 0 3  0.2664 ±0.002 0 3  0.2904 ±0.001 0 3  0.3716 ±0.002 0 3  0.4978 ±0.001 0 3
[3]  12      0.1  |   3.756 ±0.01  0 3   3.806 ±0.01  0 3  3.868  ±0.01  0 3  4.08   ±0.006 0 3  nan    ±nan   3 0  nan    ±nan   3 0  nan    ±nan   3 0  nan    ±nan   3 0
[4]  12      0.4  |   0.53  ±0.07  0 3   0.408 ±0.02  0 3  0.375  ±0.005 0 3  0.36   ±0.003 0 3  0.3732 ±0.003 0 3  0.393  ±0.003 0 3  0.4692 ±0.002 0 3  0.5884 ±0.002 0 3
[5]  12      2    |   0.8   ±0.5   0 3   0.251 ±0.008 0 3  0.242  ±0.006 0 3  0.2454 ±0.003 0 3  0.2664 ±0.002 0 3  0.2916 ±0.001 0 3  0.3744 ±0.001 0 3  0.5022 ±0.001 0 3
[6]  14      0.1  |   3.755 ±0.009 0 3   3.8   ±0.01  0 3  3.86   ±0.01  0 3  4.064  ±0.02  0 3  nan    ±nan   3 0  nan    ±nan   3 0  nan    ±nan   3 0  nan    ±nan   3 0
[7]  14      0.4  |   0.53  ±0.08  0 3   0.396 ±0.01  0 3  0.3696 ±0.004 0 3  0.3594 ±0.003 0 3  0.3702 ±0.003 0 3  0.3924 ±0.002 0 3  0.4694 ±0.001 0 3  0.5892 ±0.001 0 3
[8]  14      2    |   0.9   ±0.3   0 3   0.25  ±0.009 0 3  0.24   ±0.004 0 3  0.2448 ±0.003 0 3  0.2664 ±0.002 0 3  0.2928 ±0.002 0 3  0.3762 ±0.001 0 3  0.5052 ±0.001 0 3
"""
_print_1d(xps, "rmse.a", dict(outer="da_method",inner="infl",mean="seed",))

##
old = """[4m
•Table for N=None. •Averages Σ over ('seed',).[0m
                  |  ____infl=None____
     da_method    |  rmse.a ±1σ    ☠ ✓
---  -----------  -  -----------------
[0]  Climatology  |   3.624 ±0.006 0 3
[1]  OptInterp    |   0.941 ±0.001 0 3
[4m
•Table for N=10. •Averages Σ over ('seed',).[0m
                         |  ____infl=1.0____  _______1.01______  _______1.02______  _______1.04______  _______1.07______  _______1.1_______  _______1.2_______  _______1.4_______
     da_method  loc_rad  |  rmse.a ±1σ   ☠ ✓  rmse.a ±1σ    ☠ ✓  rmse.a ±1σ    ☠ ✓  rmse.a ±1σ    ☠ ✓  rmse.a ±1σ    ☠ ✓  rmse.a ±1σ    ☠ ✓  rmse.a ±1σ    ☠ ✓  rmse.a ±1σ    ☠ ✓
---  ---------  -------  -  ----------------  -----------------  -----------------  -----------------  -----------------  -----------------  -----------------  -----------------
[0]  EnKF                |   4.55  ±0.09 0 3   4.4   ±0.04  0 3  4.36   ±0.06  0 3  4.16   ±0.1   0 3  3.92   ±0.2   0 3  3.84   ±0.1   0 3  3.58   ±0.1   0 3  3.43   ±0.06  0 3
[1]  LETKF          0.1  |   3.758 ±0.01 0 3   3.818 ±0.01  0 3  3.892  ±0.01  0 3  4.118  ±0.01  0 3  nan    ±nan   3 0  nan    ±nan   3 0  nan    ±nan   3 0  nan    ±nan   3 0
[2]  LETKF          0.4  |   0.51  ±0.03 0 3   0.41  ±0.01  0 3  0.3752 ±0.004 0 3  0.3616 ±0.004 0 3  0.3726 ±0.003 0 3  0.3932 ±0.002 0 3  0.468  ±0.002 0 3  0.5872 ±0.001 0 3
[3]  LETKF          2    |   1.3   ±0.6  0 3   0.26  ±0.007 0 3  0.245  ±0.007 0 3  0.2454 ±0.003 0 3  0.2664 ±0.002 0 3  0.2904 ±0.001 0 3  0.3716 ±0.002 0 3  0.4978 ±0.001 0 3
[4m
•Table for N=12. •Averages Σ over ('seed',).[0m
                         |  ____infl=1.0____  _______1.01______  _______1.02______  _______1.04______  _______1.07______  _______1.1_______  _______1.2_______  _______1.4_______
     da_method  loc_rad  |  rmse.a ±1σ   ☠ ✓  rmse.a ±1σ    ☠ ✓  rmse.a ±1σ    ☠ ✓  rmse.a ±1σ    ☠ ✓  rmse.a ±1σ    ☠ ✓  rmse.a ±1σ    ☠ ✓  rmse.a ±1σ    ☠ ✓  rmse.a ±1σ    ☠ ✓
---  ---------  -------  -  ----------------  -----------------  -----------------  -----------------  -----------------  -----------------  -----------------  -----------------
[0]  EnKF                |   4.33  ±0.07 0 3   4.24  ±0.1   0 3   4.08  ±0.2   0 3  3.88   ±0.07  0 3  3.6    ±0.2   0 3  3.462  ±0.03  0 3  2.92   ±0.05  0 3  2.52   ±0.1   0 3
[1]  LETKF          0.1  |   3.756 ±0.01 0 3   3.806 ±0.01  0 3   3.868 ±0.01  0 3  4.08   ±0.006 0 3  nan    ±nan   3 0  nan    ±nan   3 0  nan    ±nan   3 0  nan    ±nan   3 0
[2]  LETKF          0.4  |   0.53  ±0.07 0 3   0.408 ±0.02  0 3   0.375 ±0.005 0 3  0.36   ±0.003 0 3  0.3732 ±0.003 0 3  0.393  ±0.003 0 3  0.4692 ±0.002 0 3  0.5884 ±0.002 0 3
[3]  LETKF          2    |   0.8   ±0.5  0 3   0.251 ±0.008 0 3   0.242 ±0.006 0 3  0.2454 ±0.003 0 3  0.2664 ±0.002 0 3  0.2916 ±0.001 0 3  0.3744 ±0.001 0 3  0.5022 ±0.001 0 3
[4m
•Table for N=14. •Averages Σ over ('seed',).[0m
                         |  _____infl=1.0____  _______1.01______  _______1.02______  _______1.04______  _______1.07______  _______1.1_______  _______1.2_______  _______1.4_______
     da_method  loc_rad  |  rmse.a ±1σ    ☠ ✓  rmse.a ±1σ    ☠ ✓  rmse.a ±1σ    ☠ ✓  rmse.a ±1σ    ☠ ✓  rmse.a ±1σ    ☠ ✓  rmse.a ±1σ    ☠ ✓  rmse.a ±1σ    ☠ ✓  rmse.a ±1σ    ☠ ✓
---  ---------  -------  -  -----------------  -----------------  -----------------  -----------------  -----------------  -----------------  -----------------  -----------------
[0]  EnKF                |   4.19  ±0.08  0 3   3.96  ±0.1   0 3  3.76   ±0.1   0 3  2.88   ±0.3   0 3  2.7    ±0.3   0 3  2.28   ±0.3   0 3  1.28   ±0.2   0 3  0.92   ±0.2   0 3
[1]  LETKF          0.1  |   3.755 ±0.009 0 3   3.8   ±0.01  0 3  3.86   ±0.01  0 3  4.064  ±0.02  0 3  nan    ±nan   3 0  nan    ±nan   3 0  nan    ±nan   3 0  nan    ±nan   3 0
[2]  LETKF          0.4  |   0.53  ±0.08  0 3   0.396 ±0.01  0 3  0.3696 ±0.004 0 3  0.3594 ±0.003 0 3  0.3702 ±0.003 0 3  0.3924 ±0.002 0 3  0.4694 ±0.001 0 3  0.5892 ±0.001 0 3
[3]  LETKF          2    |   0.9   ±0.3   0 3   0.25  ±0.009 0 3  0.24   ±0.004 0 3  0.2448 ±0.003 0 3  0.2664 ±0.002 0 3  0.2928 ±0.002 0 3  0.3762 ±0.001 0 3  0.5052 ±0.001 0 3
"""
_print_1d(xps, "rmse.a", dict(outer="N",inner="infl",mean="seed",))

##
old = """[4m
•Table for N=None. •Averages Σ over ('seed',).[0m
     |  da_method='Climatology'  ____OptInterp____
     |  rmse.a ±1σ    ☠ ✓        rmse.a ±1σ    ☠ ✓
---  -  -----------------------  -----------------
[0]  |   3.624 ±0.006 0 3         0.941 ±0.001 0 3
[4m
•Table for N=10. •Averages Σ over ('seed',).[0m
                     |  da_method='EnKF'  ______LETKF______
      infl  loc_rad  |  rmse.a ±1σ   ☠ ✓  rmse.a ±1σ    ☠ ✓
----  ----  -------  -  ----------------  -----------------
[0]   1              |  4.55   ±0.09 0 3         ±         
[1]   1         0.1  |         ±          3.758  ±0.01  0 3
[2]   1         0.4  |         ±          0.51   ±0.03  0 3
[3]   1         2    |         ±          1.3    ±0.6   0 3
[4]   1.01           |  4.4    ±0.04 0 3         ±         
[5]   1.01      0.1  |         ±          3.818  ±0.01  0 3
[6]   1.01      0.4  |         ±          0.41   ±0.01  0 3
[7]   1.01      2    |         ±          0.26   ±0.007 0 3
[8]   1.02           |  4.36   ±0.06 0 3         ±         
[9]   1.02      0.1  |         ±          3.892  ±0.01  0 3
[10]  1.02      0.4  |         ±          0.3752 ±0.004 0 3
[11]  1.02      2    |         ±          0.245  ±0.007 0 3
[12]  1.04           |  4.16   ±0.1  0 3         ±         
[13]  1.04      0.1  |         ±          4.118  ±0.01  0 3
[14]  1.04      0.4  |         ±          0.3616 ±0.004 0 3
[15]  1.04      2    |         ±          0.2454 ±0.003 0 3
[16]  1.07           |  3.92   ±0.2  0 3         ±         
[17]  1.07      0.1  |         ±          nan    ±nan   3 0
[18]  1.07      0.4  |         ±          0.3726 ±0.003 0 3
[19]  1.07      2    |         ±          0.2664 ±0.002 0 3
[20]  1.1            |  3.84   ±0.1  0 3         ±         
[21]  1.1       0.1  |         ±          nan    ±nan   3 0
[22]  1.1       0.4  |         ±          0.3932 ±0.002 0 3
[23]  1.1       2    |         ±          0.2904 ±0.001 0 3
[24]  1.2            |  3.58   ±0.1  0 3         ±         
[25]  1.2       0.1  |         ±          nan    ±nan   3 0
[26]  1.2       0.4  |         ±          0.468  ±0.002 0 3
[27]  1.2       2    |         ±          0.3716 ±0.002 0 3
[28]  1.4            |  3.43   ±0.06 0 3         ±         
[29]  1.4       0.1  |         ±          nan    ±nan   3 0
[30]  1.4       0.4  |         ±          0.5872 ±0.001 0 3
[31]  1.4       2    |         ±          0.4978 ±0.001 0 3
[4m
•Table for N=12. •Averages Σ over ('seed',).[0m
                     |  da_method='EnKF'  ______LETKF______
      infl  loc_rad  |  rmse.a ±1σ   ☠ ✓  rmse.a ±1σ    ☠ ✓
----  ----  -------  -  ----------------  -----------------
[0]   1              |  4.33   ±0.07 0 3         ±         
[1]   1         0.1  |         ±          3.756  ±0.01  0 3
[2]   1         0.4  |         ±          0.53   ±0.07  0 3
[3]   1         2    |         ±          0.8    ±0.5   0 3
[4]   1.01           |  4.24   ±0.1  0 3         ±         
[5]   1.01      0.1  |         ±          3.806  ±0.01  0 3
[6]   1.01      0.4  |         ±          0.408  ±0.02  0 3
[7]   1.01      2    |         ±          0.251  ±0.008 0 3
[8]   1.02           |  4.08   ±0.2  0 3         ±         
[9]   1.02      0.1  |         ±          3.868  ±0.01  0 3
[10]  1.02      0.4  |         ±          0.375  ±0.005 0 3
[11]  1.02      2    |         ±          0.242  ±0.006 0 3
[12]  1.04           |  3.88   ±0.07 0 3         ±         
[13]  1.04      0.1  |         ±          4.08   ±0.006 0 3
[14]  1.04      0.4  |         ±          0.36   ±0.003 0 3
[15]  1.04      2    |         ±          0.2454 ±0.003 0 3
[16]  1.07           |  3.6    ±0.2  0 3         ±         
[17]  1.07      0.1  |         ±          nan    ±nan   3 0
[18]  1.07      0.4  |         ±          0.3732 ±0.003 0 3
[19]  1.07      2    |         ±          0.2664 ±0.002 0 3
[20]  1.1            |  3.462  ±0.03 0 3         ±         
[21]  1.1       0.1  |         ±          nan    ±nan   3 0
[22]  1.1       0.4  |         ±          0.393  ±0.003 0 3
[23]  1.1       2    |         ±          0.2916 ±0.001 0 3
[24]  1.2            |  2.92   ±0.05 0 3         ±         
[25]  1.2       0.1  |         ±          nan    ±nan   3 0
[26]  1.2       0.4  |         ±          0.4692 ±0.002 0 3
[27]  1.2       2    |         ±          0.3744 ±0.001 0 3
[28]  1.4            |  2.52   ±0.1  0 3         ±         
[29]  1.4       0.1  |         ±          nan    ±nan   3 0
[30]  1.4       0.4  |         ±          0.5884 ±0.002 0 3
[31]  1.4       2    |         ±          0.5022 ±0.001 0 3
[4m
•Table for N=14. •Averages Σ over ('seed',).[0m
                     |  da_method='EnKF'  ______LETKF______
      infl  loc_rad  |  rmse.a ±1σ   ☠ ✓  rmse.a ±1σ    ☠ ✓
----  ----  -------  -  ----------------  -----------------
[0]   1              |  4.19   ±0.08 0 3         ±         
[1]   1         0.1  |         ±          3.755  ±0.009 0 3
[2]   1         0.4  |         ±          0.53   ±0.08  0 3
[3]   1         2    |         ±          0.9    ±0.3   0 3
[4]   1.01           |  3.96   ±0.1  0 3         ±         
[5]   1.01      0.1  |         ±          3.8    ±0.01  0 3
[6]   1.01      0.4  |         ±          0.396  ±0.01  0 3
[7]   1.01      2    |         ±          0.25   ±0.009 0 3
[8]   1.02           |  3.76   ±0.1  0 3         ±         
[9]   1.02      0.1  |         ±          3.86   ±0.01  0 3
[10]  1.02      0.4  |         ±          0.3696 ±0.004 0 3
[11]  1.02      2    |         ±          0.24   ±0.004 0 3
[12]  1.04           |  2.88   ±0.3  0 3         ±         
[13]  1.04      0.1  |         ±          4.064  ±0.02  0 3
[14]  1.04      0.4  |         ±          0.3594 ±0.003 0 3
[15]  1.04      2    |         ±          0.2448 ±0.003 0 3
[16]  1.07           |  2.7    ±0.3  0 3         ±         
[17]  1.07      0.1  |         ±          nan    ±nan   3 0
[18]  1.07      0.4  |         ±          0.3702 ±0.003 0 3
[19]  1.07      2    |         ±          0.2664 ±0.002 0 3
[20]  1.1            |  2.28   ±0.3  0 3         ±         
[21]  1.1       0.1  |         ±          nan    ±nan   3 0
[22]  1.1       0.4  |         ±          0.3924 ±0.002 0 3
[23]  1.1       2    |         ±          0.2928 ±0.002 0 3
[24]  1.2            |  1.28   ±0.2  0 3         ±         
[25]  1.2       0.1  |         ±          nan    ±nan   3 0
[26]  1.2       0.4  |         ±          0.4694 ±0.001 0 3
[27]  1.2       2    |         ±          0.3762 ±0.001 0 3
[28]  1.4            |  0.92   ±0.2  0 3         ±         
[29]  1.4       0.1  |         ±          nan    ±nan   3 0
[30]  1.4       0.4  |         ±          0.5892 ±0.001 0 3
[31]  1.4       2    |         ±          0.5052 ±0.001 0 3
"""
_print_1d(xps, "rmse.a", dict(outer="N",inner="da_method",mean="seed",))

##
old = """[4m
•Table for N=None.[0m
                  |  ____seed=2___  ______3______  ______4______
     da_method    |  rmse.a ±1σ     rmse.a ±1σ     rmse.a ±1σ
---  -----------  -  -------------  -------------  -------------
[0]  Climatology  |   3.632 ±0.02    3.612 ±0.02   3.628  ±0.02 
[1]  OptInterp    |   0.944 ±0.004   0.939 ±0.003  0.9396 ±0.003
[4m
•Table for N=10.[0m
                                |  ____seed=2___  ______3______  ______4______
      da_method  infl  loc_rad  |  rmse.a ±1σ     rmse.a ±1σ     rmse.a ±1σ
----  ---------  ----  -------  -  -------------  -------------  -------------
[0]   EnKF       1              |  4.54   ±0.09   4.4    ±0.1    4.72   ±0.07 
[1]   LETKF      1         0.1  |  3.776  ±0.02   3.74   ±0.02   3.76   ±0.02 
[2]   LETKF      1         0.4  |  0.54   ±0.08   0.52   ±0.05   0.456  ±0.03 
[3]   LETKF      1         2    |  2.22   ±0.3    1.32   ±0.3    0.312  ±0.02 
[4]   EnKF       1.01           |  4.48   ±0.08   4.34   ±0.1    4.37   ±0.08 
[5]   LETKF      1.01      0.1  |  3.84   ±0.02   3.804  ±0.02   3.808  ±0.02 
[6]   LETKF      1.01      0.4  |  0.416  ±0.01   0.432  ±0.04   0.386  ±0.01 
[7]   LETKF      1.01      2    |  0.275  ±0.009  0.251  ±0.008  0.254  ±0.005
[8]   EnKF       1.02           |  4.35   ±0.08   4.26   ±0.1    4.45   ±0.07 
[9]   LETKF      1.02      0.1  |  3.9    ±0.02   3.872  ±0.02   3.906  ±0.03 
[10]  LETKF      1.02      0.4  |  0.382  ±0.008  0.373  ±0.008  0.37   ±0.007
[11]  LETKF      1.02      2    |  0.257  ±0.005  0.2336 ±0.004  0.244  ±0.004
[12]  EnKF       1.04           |  4.4    ±0.07   3.92   ±0.2    4.16   ±0.09 
[13]  LETKF      1.04      0.1  |  4.136  ±0.02   4.088  ±0.02   4.128  ±0.02 
[14]  LETKF      1.04      0.4  |  0.37   ±0.006  0.355  ±0.005  0.361  ±0.005
[15]  LETKF      1.04      2    |  0.2508 ±0.003  0.2406 ±0.003  0.2436 ±0.003
[16]  EnKF       1.07           |  4.0    ±0.1    3.64   ±0.2    4.16   ±0.09 
[17]  LETKF      1.07      0.1  |  nan    ±nan    nan    ±nan    nan    ±nan  
[18]  LETKF      1.07      0.4  |  0.3776 ±0.004  0.3672 ±0.003  0.3744 ±0.003
[19]  LETKF      1.07      2    |  0.2694 ±0.003  0.2624 ±0.002  0.2664 ±0.003
[20]  EnKF       1.1            |  3.9    ±0.1    3.64   ±0.2    4.01   ±0.09 
[21]  LETKF      1.1       0.1  |  nan    ±nan    nan    ±nan    nan    ±nan  
[22]  LETKF      1.1       0.4  |  0.3944 ±0.004  0.3882 ±0.003  0.3966 ±0.003
[23]  LETKF      1.1       2    |  0.2922 ±0.003  0.2876 ±0.002  0.2916 ±0.003
[24]  EnKF       1.2            |  3.68   ±0.08   3.38   ±0.1    3.7    ±0.1  
[25]  LETKF      1.2       0.1  |  nan    ±nan    nan    ±nan    nan    ±nan  
[26]  LETKF      1.2       0.4  |  0.4698 ±0.003  0.4644 ±0.003  0.47   ±0.002
[27]  LETKF      1.2       2    |  0.3732 ±0.003  0.3684 ±0.002  0.3736 ±0.002
[28]  EnKF       1.4            |  3.5    ±0.1    3.32   ±0.1    3.48   ±0.1  
[29]  LETKF      1.4       0.1  |  nan    ±nan    nan    ±nan    nan    ±nan  
[30]  LETKF      1.4       0.4  |  0.5886 ±0.003  0.585  ±0.003  0.588  ±0.002
[31]  LETKF      1.4       2    |  0.4992 ±0.003  0.495  ±0.003  0.4992 ±0.002
[4m
•Table for N=12.[0m
                                |  ____seed=2___  ______3______  ______4______
      da_method  infl  loc_rad  |  rmse.a ±1σ     rmse.a ±1σ     rmse.a ±1σ
----  ---------  ----  -------  -  -------------  -------------  -------------
[0]   EnKF       1              |  4.26   ±0.1    4.24   ±0.2    4.47   ±0.07 
[1]   LETKF      1         0.1  |  3.776  ±0.02   3.74   ±0.02   3.756  ±0.02 
[2]   LETKF      1         0.4  |  0.492  ±0.03   0.68   ±0.1    0.428  ±0.02 
[3]   LETKF      1         2    |  1.7    ±0.6    0.304  ±0.02   0.324  ±0.03 
[4]   EnKF       1.01           |  4.41   ±0.09   4.04   ±0.2    4.27   ±0.06 
[5]   LETKF      1.01      0.1  |  3.824  ±0.02   3.784  ±0.02   3.812  ±0.02 
[6]   LETKF      1.01      0.4  |  0.416  ±0.02   0.43   ±0.05   0.376  ±0.009
[7]   LETKF      1.01      2    |  0.266  ±0.007  0.2408 ±0.004  0.248  ±0.005
[8]   EnKF       1.02           |  4.24   ±0.08   3.76   ±0.2    4.28   ±0.07 
[9]   LETKF      1.02      0.1  |  3.888  ±0.02   3.84   ±0.02   3.872  ±0.02 
[10]  LETKF      1.02      0.4  |  0.384  ±0.008  0.376  ±0.01   0.367  ±0.007
[11]  LETKF      1.02      2    |  0.254  ±0.006  0.234  ±0.003  0.2392 ±0.004
[12]  EnKF       1.04           |  3.94   ±0.09   3.74   ±0.1    3.94   ±0.1  
[13]  LETKF      1.04      0.1  |  4.088  ±0.02   4.08   ±0.02   4.068  ±0.02 
[14]  LETKF      1.04      0.4  |  0.365  ±0.005  0.354  ±0.005  0.3608 ±0.004
[15]  LETKF      1.04      2    |  0.2512 ±0.004  0.2406 ±0.003  0.2442 ±0.003
[16]  EnKF       1.07           |  3.8    ±0.1    3.3    ±0.3    3.72   ±0.1  
[17]  LETKF      1.07      0.1  |  nan    ±nan    nan    ±nan    nan    ±nan  
[18]  LETKF      1.07      0.4  |  0.376  ±0.004  0.3672 ±0.004  0.3756 ±0.003
[19]  LETKF      1.07      2    |  0.2694 ±0.003  0.2632 ±0.002  0.267  ±0.003
[20]  EnKF       1.1            |  3.44   ±0.2    3.44   ±0.2    3.52   ±0.1  
[21]  LETKF      1.1       0.1  |  nan    ±nan    nan    ±nan    nan    ±nan  
[22]  LETKF      1.1       0.4  |  0.396  ±0.003  0.3882 ±0.003  0.396  ±0.003
[23]  LETKF      1.1       2    |  0.2934 ±0.003  0.2888 ±0.002  0.2928 ±0.002
[24]  EnKF       1.2            |  2.96   ±0.2    2.84   ±0.2    2.96   ±0.2  
[25]  LETKF      1.2       0.1  |  nan    ±nan    nan    ±nan    nan    ±nan  
[26]  LETKF      1.2       0.4  |  0.471  ±0.003  0.4656 ±0.003  0.4712 ±0.002
[27]  LETKF      1.2       2    |  0.3756 ±0.003  0.3712 ±0.002  0.376  ±0.002
[28]  EnKF       1.4            |  2.34   ±0.3    2.48   ±0.2    2.76   ±0.2  
[29]  LETKF      1.4       0.1  |  nan    ±nan    nan    ±nan    nan    ±nan  
[30]  LETKF      1.4       0.4  |  0.5898 ±0.003  0.585  ±0.003  0.59   ±0.002
[31]  LETKF      1.4       2    |  0.5034 ±0.003  0.4998 ±0.003  0.5036 ±0.002
[4m
•Table for N=14.[0m
                                |  ____seed=2___  ______3______  ______4______
      da_method  infl  loc_rad  |  rmse.a ±1σ     rmse.a ±1σ     rmse.a ±1σ
----  ---------  ----  -------  -  -------------  -------------  -------------
[0]   EnKF       1              |  4.08   ±0.2    4.12   ±0.2    4.36   ±0.1  
[1]   LETKF      1         0.1  |  3.764  ±0.02   3.736  ±0.02   3.764  ±0.02 
[2]   LETKF      1         0.4  |  0.464  ±0.04   0.68   ±0.1    0.424  ±0.02 
[3]   LETKF      1         2    |  1.04   ±0.4    0.292  ±0.01   1.4    ±0.6  
[4]   EnKF       1.01           |  4.04   ±0.2    3.68   ±0.2    4.16   ±0.2  
[5]   LETKF      1.01      0.1  |  3.816  ±0.02   3.776  ±0.02   3.804  ±0.02 
[6]   LETKF      1.01      0.4  |  0.412  ±0.02   0.396  ±0.01   0.376  ±0.008
[7]   LETKF      1.01      2    |  0.267  ±0.008  0.2352 ±0.004  0.249  ±0.005
[8]   EnKF       1.02           |  3.92   ±0.2    3.88   ±0.2    3.5    ±0.5  
[9]   LETKF      1.02      0.1  |  3.868  ±0.02   3.836  ±0.02   3.872  ±0.02 
[10]  LETKF      1.02      0.4  |  0.378  ±0.008  0.368  ±0.01   0.365  ±0.006
[11]  LETKF      1.02      2    |  0.2472 ±0.004  0.2334 ±0.003  0.2392 ±0.004
[12]  EnKF       1.04           |  3.4    ±0.2    2.5    ±0.6    2.6    ±0.6  
[13]  LETKF      1.04      0.1  |  4.088  ±0.02   4.028  ±0.02   4.08   ±0.03 
[14]  LETKF      1.04      0.4  |  0.365  ±0.005  0.354  ±0.005  0.3592 ±0.004
[15]  LETKF      1.04      2    |  0.2496 ±0.004  0.2406 ±0.003  0.2448 ±0.003
[16]  EnKF       1.07           |  3.16   ±0.2    2.82   ±0.3    2.2    ±0.6  
[17]  LETKF      1.07      0.1  |  nan    ±nan    nan    ±nan    nan    ±nan  
[18]  LETKF      1.07      0.4  |  0.3744 ±0.004  0.3642 ±0.003  0.3732 ±0.003
[19]  LETKF      1.07      2    |  0.2688 ±0.003  0.2628 ±0.002  0.2682 ±0.003
[20]  EnKF       1.1            |  2.76   ±0.3    2.34   ±0.3    1.6    ±0.5  
[21]  LETKF      1.1       0.1  |  nan    ±nan    nan    ±nan    nan    ±nan  
[22]  LETKF      1.1       0.4  |  0.3954 ±0.003  0.3876 ±0.003  0.3948 ±0.003
[23]  LETKF      1.1       2    |  0.2946 ±0.003  0.2892 ±0.002  0.294  ±0.003
[24]  EnKF       1.2            |  1.28   ±0.4    0.96   ±0.3    1.62   ±0.3  
[25]  LETKF      1.2       0.1  |  nan    ±nan    nan    ±nan    nan    ±nan  
[26]  LETKF      1.2       0.4  |  0.4704 ±0.003  0.4662 ±0.003  0.4712 ±0.002
[27]  LETKF      1.2       2    |  0.3774 ±0.003  0.3736 ±0.002  0.3776 ±0.002
[28]  EnKF       1.4            |  1.16   ±0.2    0.55   ±0.05   1.08   ±0.3  
[29]  LETKF      1.4       0.1  |  nan    ±nan    nan    ±nan    nan    ±nan  
[30]  LETKF      1.4       0.4  |  0.5904 ±0.003  0.5868 ±0.003  0.5908 ±0.002
[31]  LETKF      1.4       2    |  0.507  ±0.003  0.5028 ±0.003  0.506  ±0.002
"""
_print_1d(xps, "rmse.a", dict(outer="N",inner="seed"))

##
old = """[4m
•Table for da_method='Climatology'.[0m
           |  ___N=None___
     seed  |  rmse.a ±1σ
---  ----  -  ------------
[0]     2  |   3.632 ±0.02
[1]     3  |   3.612 ±0.02
[2]     4  |   3.628 ±0.02
[4m
•Table for da_method='OptInterp'.[0m
           |  ____N=None___
     seed  |  rmse.a ±1σ
---  ----  -  -------------
[0]     2  |  0.944  ±0.004
[1]     3  |  0.939  ±0.003
[2]     4  |  0.9396 ±0.003
[4m
•Table for da_method='EnKF'.[0m
                  |  ____N=10____  _____12_____  _____14_____
      seed  infl  |  rmse.a ±1σ    rmse.a ±1σ    rmse.a ±1σ
----  ----  ----  -  ------------  ------------  ------------
[0]      2  1     |    4.54 ±0.09    4.26 ±0.1     4.08 ±0.2 
[1]      2  1.01  |    4.48 ±0.08    4.41 ±0.09    4.04 ±0.2 
[2]      2  1.02  |    4.35 ±0.08    4.24 ±0.08    3.92 ±0.2 
[3]      2  1.04  |    4.4  ±0.07    3.94 ±0.09    3.4  ±0.2 
[4]      2  1.07  |    4    ±0.1     3.8  ±0.1     3.16 ±0.2 
[5]      2  1.1   |    3.9  ±0.1     3.44 ±0.2     2.76 ±0.3 
[6]      2  1.2   |    3.68 ±0.08    2.96 ±0.2     1.28 ±0.4 
[7]      2  1.4   |    3.5  ±0.1     2.34 ±0.3     1.16 ±0.2 
[8]      3  1     |    4.4  ±0.1     4.24 ±0.2     4.12 ±0.2 
[9]      3  1.01  |    4.34 ±0.1     4.04 ±0.2     3.68 ±0.2 
[10]     3  1.02  |    4.26 ±0.1     3.76 ±0.2     3.88 ±0.2 
[11]     3  1.04  |    3.92 ±0.2     3.74 ±0.1     2.5  ±0.6 
[12]     3  1.07  |    3.64 ±0.2     3.3  ±0.3     2.82 ±0.3 
[13]     3  1.1   |    3.64 ±0.2     3.44 ±0.2     2.34 ±0.3 
[14]     3  1.2   |    3.38 ±0.1     2.84 ±0.2     0.96 ±0.3 
[15]     3  1.4   |    3.32 ±0.1     2.48 ±0.2     0.55 ±0.05
[16]     4  1     |    4.72 ±0.07    4.47 ±0.07    4.36 ±0.1 
[17]     4  1.01  |    4.37 ±0.08    4.27 ±0.06    4.16 ±0.2 
[18]     4  1.02  |    4.45 ±0.07    4.28 ±0.07    3.5  ±0.5 
[19]     4  1.04  |    4.16 ±0.09    3.94 ±0.1     2.6  ±0.6 
[20]     4  1.07  |    4.16 ±0.09    3.72 ±0.1     2.2  ±0.6 
[21]     4  1.1   |    4.01 ±0.09    3.52 ±0.1     1.6  ±0.5 
[22]     4  1.2   |    3.7  ±0.1     2.96 ±0.2     1.62 ±0.3 
[23]     4  1.4   |    3.48 ±0.1     2.76 ±0.2     1.08 ±0.3 
"""
_print_1d(xps_shorter, "rmse.a", dict(outer="da_method",inner="N"))

##
old = """[4m
•Table for da_method='Climatology'.[0m
           |  _____N=None____
     seed  |  rmse.a ±1σ  ☠ ✓
---  ----  -  ---------------
[0]     2  |   3.632 ±nan 0 1
[1]     3  |   3.611 ±nan 0 1
[2]     4  |   3.628 ±nan 0 1
[4m
•Table for da_method='OptInterp'.[0m
           |  _____N=None____
     seed  |  rmse.a ±1σ  ☠ ✓
---  ----  -  ---------------
[0]     2  |  0.9436 ±nan 0 1
[1]     3  |  0.9393 ±nan 0 1
[2]     4  |  0.9398 ±nan 0 1
[4m
•Table for da_method='EnKF'.[0m
                  |  ______N=10_____  _______12______  _______14______
      seed  infl  |  rmse.a ±1σ  ☠ ✓  rmse.a ±1σ  ☠ ✓  rmse.a ±1σ  ☠ ✓
----  ----  ----  -  ---------------  ---------------  ---------------
[0]      2  1     |   4.531 ±nan 0 1   4.259 ±nan 0 1  4.098  ±nan 0 1
[1]      2  1.01  |   4.479 ±nan 0 1   4.406 ±nan 0 1  4.043  ±nan 0 1
[2]      2  1.02  |   4.351 ±nan 0 1   4.241 ±nan 0 1  3.911  ±nan 0 1
[3]      2  1.04  |   4.391 ±nan 0 1   3.945 ±nan 0 1  3.411  ±nan 0 1
[4]      2  1.07  |   3.997 ±nan 0 1   3.794 ±nan 0 1  3.157  ±nan 0 1
[5]      2  1.1   |   3.904 ±nan 0 1   3.429 ±nan 0 1  2.774  ±nan 0 1
[6]      2  1.2   |   3.68  ±nan 0 1   2.97  ±nan 0 1  1.273  ±nan 0 1
[7]      2  1.4   |   3.508 ±nan 0 1   2.335 ±nan 0 1  1.166  ±nan 0 1
[8]      3  1     |   4.393 ±nan 0 1   4.238 ±nan 0 1  4.119  ±nan 0 1
[9]      3  1.01  |   4.347 ±nan 0 1   4.041 ±nan 0 1  3.695  ±nan 0 1
[10]     3  1.02  |   4.254 ±nan 0 1   3.772 ±nan 0 1  3.884  ±nan 0 1
[11]     3  1.04  |   3.929 ±nan 0 1   3.734 ±nan 0 1  2.556  ±nan 0 1
[12]     3  1.07  |   3.626 ±nan 0 1   3.289 ±nan 0 1  2.839  ±nan 0 1
[13]     3  1.1   |   3.624 ±nan 0 1   3.444 ±nan 0 1  2.365  ±nan 0 1
[14]     3  1.2   |   3.377 ±nan 0 1   2.823 ±nan 0 1  0.9706 ±nan 0 1
[15]     3  1.4   |   3.316 ±nan 0 1   2.47  ±nan 0 1  0.5529 ±nan 0 1
[16]     4  1     |   4.712 ±nan 0 1   4.468 ±nan 0 1  4.356  ±nan 0 1
[17]     4  1.01  |   4.369 ±nan 0 1   4.266 ±nan 0 1  4.163  ±nan 0 1
[18]     4  1.02  |   4.454 ±nan 0 1   4.284 ±nan 0 1  3.49   ±nan 0 1
[19]     4  1.04  |   4.166 ±nan 0 1   3.939 ±nan 0 1  2.634  ±nan 0 1
[20]     4  1.07  |   4.166 ±nan 0 1   3.723 ±nan 0 1  2.162  ±nan 0 1
[21]     4  1.1   |   4.019 ±nan 0 1   3.511 ±nan 0 1  1.65   ±nan 0 1
[22]     4  1.2   |   3.707 ±nan 0 1   2.979 ±nan 0 1  1.6    ±nan 0 1
[23]     4  1.4   |   3.478 ±nan 0 1   2.776 ±nan 0 1  1.051  ±nan 0 1
"""
_print_1d(xps_shorter, "rmse.a", dict(outer="da_method",inner="N",mean=()))

##
old = """[4m
•Table for da_method='Climatology'. •Averages Σ over ('seed', 'infl').[0m
     |  ______N=None_____
     |  rmse.a ±1σ    ☠ ✓
---  -  -----------------
[0]  |   3.624 ±0.006 0 3
[4m
•Table for da_method='OptInterp'. •Averages Σ over ('seed', 'infl').[0m
     |  ______N=None_____
     |  rmse.a ±1σ    ☠ ✓
---  -  -----------------
[0]  |   0.941 ±0.001 0 3
[4m
•Table for da_method='EnKF'. •Averages Σ over ('seed', 'infl').[0m
     |  _______N=10______  _______12_______  _______14_______
     |  rmse.a ±1σ   ☠  ✓  rmse.a ±1σ  ☠  ✓  rmse.a ±1σ  ☠  ✓
---  -  -----------------  ----------------  ----------------
[0]  |    4.03 ±0.08 0 24    3.64 ±0.1 0 24    2.76 ±0.2 0 24
"""
_print_1d(xps_shorter, "rmse.a", dict(outer="da_method",inner="N",mean=("seed","infl")))

##
old = """[4m
•Table for da_method='Climatology'. •Averages Σ over ('seed',).[0m
     |  rmse.a ±1σ    ☠ ✓
---  -  -----------------
[0]  |   3.624 ±0.006 0 3
[4m
•Table for da_method='OptInterp'. •Averages Σ over ('seed',).[0m
     |  rmse.a ±1σ    ☠ ✓
---  -  -----------------
[0]  |   0.941 ±0.001 0 3
[4m
•Table for da_method='EnKF'. •Averages Σ over ('seed',).[0m
       N  infl  |  rmse.a ±1σ   ☠ ✓
----  --  ----  -  ----------------
[0]   10  1     |   4.55  ±0.09 0 3
[1]   10  1.01  |   4.4   ±0.04 0 3
[2]   10  1.02  |   4.36  ±0.06 0 3
[3]   10  1.04  |   4.16  ±0.1  0 3
[4]   10  1.07  |   3.92  ±0.2  0 3
[5]   10  1.1   |   3.84  ±0.1  0 3
[6]   10  1.2   |   3.58  ±0.1  0 3
[7]   10  1.4   |   3.43  ±0.06 0 3
[8]   12  1     |   4.33  ±0.07 0 3
[9]   12  1.01  |   4.24  ±0.1  0 3
[10]  12  1.02  |   4.08  ±0.2  0 3
[11]  12  1.04  |   3.88  ±0.07 0 3
[12]  12  1.07  |   3.6   ±0.2  0 3
[13]  12  1.1   |   3.462 ±0.03 0 3
[14]  12  1.2   |   2.92  ±0.05 0 3
[15]  12  1.4   |   2.52  ±0.1  0 3
[16]  14  1     |   4.19  ±0.08 0 3
[17]  14  1.01  |   3.96  ±0.1  0 3
[18]  14  1.02  |   3.76  ±0.1  0 3
[19]  14  1.04  |   2.88  ±0.3  0 3
[20]  14  1.07  |   2.7   ±0.3  0 3
[21]  14  1.1   |   2.28  ±0.3  0 3
[22]  14  1.2   |   1.28  ±0.2  0 3
[23]  14  1.4   |   0.92  ±0.2  0 3
"""
_print_1d(xps_shorter, "rmse.a", dict(outer="da_method",mean=("seed")))

##
old = """[4m
•Table for da_method='Climatology'.[0m
     seed  |  rmse.a ±1σ
---  ----  -  ------------
[0]     2  |   3.632 ±0.02
[1]     3  |   3.612 ±0.02
[2]     4  |   3.628 ±0.02
[4m
•Table for da_method='OptInterp'.[0m
     seed  |  rmse.a ±1σ
---  ----  -  -------------
[0]     2  |  0.944  ±0.004
[1]     3  |  0.939  ±0.003
[2]     4  |  0.9396 ±0.003
[4m
•Table for da_method='EnKF'.[0m
      seed   N  infl  |  rmse.a ±1σ
----  ----  --  ----  -  ------------
[0]      2  10  1     |    4.54 ±0.09
[1]      2  10  1.01  |    4.48 ±0.08
[2]      2  10  1.02  |    4.35 ±0.08
[3]      2  10  1.04  |    4.4  ±0.07
[4]      2  10  1.07  |    4    ±0.1 
[5]      2  10  1.1   |    3.9  ±0.1 
[6]      2  10  1.2   |    3.68 ±0.08
[7]      2  10  1.4   |    3.5  ±0.1 
[8]      2  12  1     |    4.26 ±0.1 
[9]      2  12  1.01  |    4.41 ±0.09
[10]     2  12  1.02  |    4.24 ±0.08
[11]     2  12  1.04  |    3.94 ±0.09
[12]     2  12  1.07  |    3.8  ±0.1 
[13]     2  12  1.1   |    3.44 ±0.2 
[14]     2  12  1.2   |    2.96 ±0.2 
[15]     2  12  1.4   |    2.34 ±0.3 
[16]     2  14  1     |    4.08 ±0.2 
[17]     2  14  1.01  |    4.04 ±0.2 
[18]     2  14  1.02  |    3.92 ±0.2 
[19]     2  14  1.04  |    3.4  ±0.2 
[20]     2  14  1.07  |    3.16 ±0.2 
[21]     2  14  1.1   |    2.76 ±0.3 
[22]     2  14  1.2   |    1.28 ±0.4 
[23]     2  14  1.4   |    1.16 ±0.2 
[24]     3  10  1     |    4.4  ±0.1 
[25]     3  10  1.01  |    4.34 ±0.1 
[26]     3  10  1.02  |    4.26 ±0.1 
[27]     3  10  1.04  |    3.92 ±0.2 
[28]     3  10  1.07  |    3.64 ±0.2 
[29]     3  10  1.1   |    3.64 ±0.2 
[30]     3  10  1.2   |    3.38 ±0.1 
[31]     3  10  1.4   |    3.32 ±0.1 
[32]     3  12  1     |    4.24 ±0.2 
[33]     3  12  1.01  |    4.04 ±0.2 
[34]     3  12  1.02  |    3.76 ±0.2 
[35]     3  12  1.04  |    3.74 ±0.1 
[36]     3  12  1.07  |    3.3  ±0.3 
[37]     3  12  1.1   |    3.44 ±0.2 
[38]     3  12  1.2   |    2.84 ±0.2 
[39]     3  12  1.4   |    2.48 ±0.2 
[40]     3  14  1     |    4.12 ±0.2 
[41]     3  14  1.01  |    3.68 ±0.2 
[42]     3  14  1.02  |    3.88 ±0.2 
[43]     3  14  1.04  |    2.5  ±0.6 
[44]     3  14  1.07  |    2.82 ±0.3 
[45]     3  14  1.1   |    2.34 ±0.3 
[46]     3  14  1.2   |    0.96 ±0.3 
[47]     3  14  1.4   |    0.55 ±0.05
[48]     4  10  1     |    4.72 ±0.07
[49]     4  10  1.01  |    4.37 ±0.08
[50]     4  10  1.02  |    4.45 ±0.07
[51]     4  10  1.04  |    4.16 ±0.09
[52]     4  10  1.07  |    4.16 ±0.09
[53]     4  10  1.1   |    4.01 ±0.09
[54]     4  10  1.2   |    3.7  ±0.1 
[55]     4  10  1.4   |    3.48 ±0.1 
[56]     4  12  1     |    4.47 ±0.07
[57]     4  12  1.01  |    4.27 ±0.06
[58]     4  12  1.02  |    4.28 ±0.07
[59]     4  12  1.04  |    3.94 ±0.1 
[60]     4  12  1.07  |    3.72 ±0.1 
[61]     4  12  1.1   |    3.52 ±0.1 
[62]     4  12  1.2   |    2.96 ±0.2 
[63]     4  12  1.4   |    2.76 ±0.2 
[64]     4  14  1     |    4.36 ±0.1 
[65]     4  14  1.01  |    4.16 ±0.2 
[66]     4  14  1.02  |    3.5  ±0.5 
[67]     4  14  1.04  |    2.6  ±0.6 
[68]     4  14  1.07  |    2.2  ±0.6 
[69]     4  14  1.1   |    1.6  ±0.5 
[70]     4  14  1.2   |    1.62 ±0.3 
[71]     4  14  1.4   |    1.08 ±0.3 
"""
_print_1d(xps_shorter, "rmse.a", dict(outer="da_method"))

##
old = """           |  da_method='Climatology', N=None  OptInterp, None  EnKF, 10  EnKF, 12  EnKF, 14
     infl  |  rmse.a                           rmse.a           rmse.a    rmse.a    rmse.a
---  ----  -  -------------------------------  ---------------  --------  --------  --------
[0]        |  3.624                            0.941                                      
[1]  1     |                                                    4.55      4.33      4.19  
[2]  1.01  |                                                    4.4       4.24      3.96  
[3]  1.02  |                                                    4.36      4.08      3.76  
[4]  1.04  |                                                    4.16      3.88      2.88  
[5]  1.07  |                                                    3.92      3.6       2.7   
[6]  1.1   |                                                    3.84      3.462     2.28  
[7]  1.2   |                                                    3.58      2.92      1.28  
[8]  1.4   |                                                    3.43      2.52      0.92  
"""
_print_1d(xps_shorter, "rmse.a", dict(inner=("da_method","N"),mean="seed"), subcols=False)

##
old = """[4m
•Table for da_method='Climatology', seed=2.[0m
     |  N=None
     |  rmse.a
---  -  ------
[0]  |   3.632
[4m
•Table for da_method='OptInterp', seed=2.[0m
     |  N=None
     |  rmse.a
---  -  ------
[0]  |   0.944
[4m
•Table for da_method='EnKF', seed=2.[0m
           |  _N=10_  __12__  __14__
     infl  |  rmse.a  rmse.a  rmse.a
---  ----  -  ------  ------  ------
[0]  1     |    4.54    4.26    4.08
[1]  1.01  |    4.48    4.41    4.04
[2]  1.02  |    4.35    4.24    3.92
[3]  1.04  |    4.4     3.94    3.4 
[4]  1.07  |    4       3.8     3.16
[5]  1.1   |    3.9     3.44    2.76
[6]  1.2   |    3.68    2.96    1.28
[7]  1.4   |    3.5     2.34    1.16
[4m
•Table for da_method='Climatology', seed=3.[0m
     |  N=None
     |  rmse.a
---  -  ------
[0]  |   3.612
[4m
•Table for da_method='OptInterp', seed=3.[0m
     |  N=None
     |  rmse.a
---  -  ------
[0]  |   0.939
[4m
•Table for da_method='EnKF', seed=3.[0m
           |  _N=10_  __12__  __14__
     infl  |  rmse.a  rmse.a  rmse.a
---  ----  -  ------  ------  ------
[0]  1     |    4.4     4.24    4.12
[1]  1.01  |    4.34    4.04    3.68
[2]  1.02  |    4.26    3.76    3.88
[3]  1.04  |    3.92    3.74    2.5 
[4]  1.07  |    3.64    3.3     2.82
[5]  1.1   |    3.64    3.44    2.34
[6]  1.2   |    3.38    2.84    0.96
[7]  1.4   |    3.32    2.48    0.55
[4m
•Table for da_method='Climatology', seed=4.[0m
     |  N=None
     |  rmse.a
---  -  ------
[0]  |   3.628
[4m
•Table for da_method='OptInterp', seed=4.[0m
     |  N=None
     |  rmse.a
---  -  ------
[0]  |  0.9396
[4m
•Table for da_method='EnKF', seed=4.[0m
           |  _N=10_  __12__  __14__
     infl  |  rmse.a  rmse.a  rmse.a
---  ----  -  ------  ------  ------
[0]  1     |    4.72    4.47    4.36
[1]  1.01  |    4.37    4.27    4.16
[2]  1.02  |    4.45    4.28    3.5 
[3]  1.04  |    4.16    3.94    2.6 
[4]  1.07  |    4.16    3.72    2.2 
[5]  1.1   |    4.01    3.52    1.6 
[6]  1.2   |    3.7     2.96    1.62
[7]  1.4   |    3.48    2.76    1.08
"""
_print_1d(xps_shorter, "rmse.a", dict(outer=("da_method","seed"),inner="N"), subcols=False)


##
if "--replace" in sys.argv:
    new_code = orig_code[0:replacements[0].nOpen]

    for i,replacement in enumerate(replacements):

        replacement.lines[0] = 'old = """' + replacement.lines[0]
        new_code += replacement.lines

        try:
            nEnd = replacements[i+1].nOpen
        except IndexError:
            nEnd = len(orig_code)
        new_code += orig_code[replacement.nClose : nEnd]

    # Don't overwrite! This allows for diffing.
    with open(__file__+".new", "w") as F:
        for line in new_code:
            # F.write(line.rstrip()+"\n")
            F.write(line)

##

##

def plot1d(hypercube, statkey="rmse.a",
        axes=_DEFAULT_ALLOTMENT,
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
           if flexcomp(k, 'da_method', 'single_out_tag'):
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
        axis = hypercube.axis_ticks_nn(marker_axis)
        if index in [None, -1]:   return '.'
        else:                     return markers[index%len(markers)]
    def _linestyle(index):
        axis = hypercube.axis_ticks_nn(linestyle_axis)
        if index in [None, -1]:   return '-'
        else:                     return linestyles[index%len(linestyles)]
    def _alpha(index):
        axis = hypercube.axis_ticks_nn(alpha_axis)
        if   index in [None, -1]: return 1
        else:                     return ((1+index)/len(axis))**1.5
    def _color(index):
        axis = hypercube.axis_ticks_nn(color_axis)
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
    axes, tables = nest_xps(hypercube,statkey,axes)
    xticks = hypercube.axis_ticks_nn(axes["inner"][0])

    # Validate axes
    assert len(axes["inner"]) == 1, "You must chose the abscissa."
    for ak in style_dict:
        av = style_dict[ak]['axis']
        assert av in hypercube.axes, f"Axis {av!r} not among hypercube.axes."
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
    for ip, table_coord in enumerate(tables):
        table = tables[table_coord]
        title = '' if axes["outer"] is None else table_coord.str_dict()
        table.panels = panels[:,ip]

        # Plot
        for coord in table.rows:
            row = table.rows[coord]
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
            axis = hypercube.axis_ticks_nn(a)
            if isinstance(axis[0], bool):
                ylims = 0, 1
            else:
                ylims = axis[0], axis[-1]
            ax.set_ylim(*stretch(*ylims,1.02))
            if ax.is_first_col():
                ax.set_ylabel(f"Optim.\n{a}")

    return tables

##

savepath = '/home/pnr/dpr_data/example_3/run_2020-01-02_00-00-00'

# The following **only** uses saved data => Can run as a separate script.
xps = load_xps(savepath)

# Remove experiments we don't want to plot:
xps = [xp for xp in xps if True
    and getattr(xp,'upd_a'    ,None)!="PertObs"
    and getattr(xp,'da_method',None)!="EnKF_N"
    and getattr(xp,'HMM_F')         !=10
    ]

# Associate each control variable with a dimension in "hyperspace"
xps = ExperimentHypercube.from_list(xps)

# Single out a few particular experiment types to add to plot
xps.single_out(dict(da_method='EnKF' ,infl=1.01), 'NO-infl NO-loc' , ('infl'))
xps.single_out(dict(da_method='LETKF',infl=1.01), 'NO-infl'        , ('infl'))

##

def beautify_figure(tabulated_data):
    """Beautify.

    These settings are particular to a certain type of plots,
    and so do not generalize well.
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
    axs = array([col.panels for col in tabulated_data.as_list()]).T

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
        if not ax.is_first_row() and 'infl' in ax.get_ylabel():
            yy = xps.axis_ticks_nn('infl')
            axis_scale_by_array(ax, yy, "y")

# Try mixing around the various axes allotments:
plt.ion()
tabulated_data = plot1d(xps, 'rmse.a', fignum=1,
        axes=dict(
            inner="N",
            # outer="seed",
            mean="seed",
            optim=('loc_rad','infl'),
            ),
    #     marker_axis="da_method", color_axis='infl', color_in_legend=False, 
    #     marker_axis='seed',                        marker_in_legend=False,
    #  linestyle_axis="rot",                      linestyle_in_legend=True,
    )
beautify_figure(tabulated_data)

##
tabulated_data = plot1d(xps, 'rmse.a', fignum=1,
        axes=dict(
            inner='infl', 
            outer='da_method',
            mean=('seed',),
            optim=('loc_rad'),
            ),
         color_axis='N', color_in_legend=False, 
        # marker_axis='da_method',
        )
beautify_figure(tabulated_data)

##
tabulated_data = plot1d(xps, 'rmse.a', fignum=1,
        axes=dict(
            inner='N',
            outer="infl",
            mean=('seed',),
            optim=('loc_rad'),
            )
        )

beautify_figure(tabulated_data)

##
