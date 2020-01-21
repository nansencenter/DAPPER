"""Test data loading and presenting functionality."""
# TODO:
# - Make group_along return xpCube
#   - Update plot_1d and print_1d to take advantage of this
#   - Fix the issue that print_1d(xps,mean_axes=None,panel_axes=())
#     hides the da_methods that wouldn't appear in 1st column (N=10),
#     [because that's when it makes its distinct attr list].
#     This can be fixed by getting distinct from
#     cube2 = ExperimentHypercube(xps_group).group_along("N")
#     distinct, _, _ = cube2.xp_list.split_attrs()
# - Include tuning_axes in print_1d
# - Fix ax vs axis vs axes
# - Rename xp_list to as_list
# - Rename xp_dict to as_dict

# In print_1d:
# - replace all np.vectorize in math.py with vectorize0
# - replace header argument in tabulate_column with minWidth.



##
from dapper import *
##
def unpack_uqs(lst, get_counts=False, decimals=None):

    subcolumns = ["val","conf"]
    if get_counts:
        subcolumns += ["nFail","nSuccess"]

    # np.array with named columns.
    dtype = object # enables storing None's (not as nan's)
    dtype = np.dtype([(c,dtype) for c in subcolumns])
    avrgs = np.full_like(lst, dtype=dtype, fill_value=None)

    for i,uq in enumerate(lst):
        if uq is not None:
            if decimals is None: v,c = uq.round(mult=0.2)
            else:                v,c = np.round([uq.val, uq.conf],decimals)

            with set_tmp(uq,'val',v), set_tmp(uq,'conf',c):
                for c in subcolumns:
                    avrgs[c][i] = getattr(uq,c)

    return avrgs


def print_1d(hypercube, statkey="rmse.a", panel_axes=("da_method",),
        column_axis="N",
        mean_axes=("seed",),
        ):

    # Used many times ==> abbreviate
    mn = mean_axes is not None

    # Define columns
    if column_axis is None:
        col_range = [None]
    elif isinstance(column_axis, str):
        col_range = hypercube.axes[column_axis]

    def get_col(xps,ind):
        if column_axis is None:
            return xps # return full list of xps
        else:
            return xps[xps.inds(missingval=None,
                **{column_axis:col_range[ind]})]

    group_axes = complement(hypercube.axes, *panel_axes)
    groups = hypercube.group_along(*group_axes)

    for group_coord, inds_in_group in groups.items():

        table_title = ", ".join([f"{k}={v}" for k,v in
            group_coord._asdict().items() if k in panel_axes])

        xps_group = hypercube[inds_in_group]
        init_done = False

        # Make columns
        for j,col_val in enumerate(col_range):
            xps_col = get_col(xps_group,j)
            if not xps_col: continue

            if mn:
                # Average
                cube_group = ExperimentHypercube(xps_col)
                mean_axes_col = intersect(mean_axes, *cube_group.axes)
                # NB: don't use set() intersection -- keep ordering for pytest.
                mn_title = f"Stats avrg'd wrt. {mean_axes_col}."
                mean_cube = cube_group.mean_field(statkey, mean_axes_col)
                # Limit xps_col to seed0
                seed0 = {ax:hypercube.axes[ax][0] for ax in mean_axes_col}
                xps_col = xps_col[xps_col.inds(**seed0)]

            if not init_done:
                init_done = True
                print("\n" + table_title)
                if mn: print(mn_title)
                # Make attribute (left) part of table
                distinct, redundant, common = xps_col.split_attrs()
                headers = list(distinct.keys()) + ['|' + '\n|' if column_axis else '']
                matters = list(distinct.values()) + [['|']*len(xps_col)]

            # Collect avrgs for this column
            avrgs = []
            # NB: Don't do: avrgs = [xp.avrgs for xp in xps_col]
            # coz there's no guarantee that it be ordered the same as distinct.
            for row in range(len(matters[0])):
                attrs = {a: distinct[a][row] for a in distinct}

                if mn:
                    coord = cube_group.make_coord(**attrs,
                            **{k:"NULL" for k in mean_axes_col})
                    avrgs += [mean_cube[coord]]
                else:
                    xp = xps_col[xps_col.inds(**attrs)]
                    sk = de_abbrev(statkey)
                    avrg = lambda xp: deep_getattr(xp,f'avrgs.{sk}',None)
                    avrgs += [avrg(xp[0]) if xp else None]


            column = unpack_uqs(avrgs, get_counts=mn)

            # Tabulate each subcolumn:
            for subcol in column.dtype.names:
                header = statkey if subcol is "val" else ""
                column[subcol] = tabulate_column(column[subcol],header,'æ')[1:]

            # Join subcolumns:
            if mn: column_template = '{} ±{} {} {}' ; spaces = ['  ',' ','']
            else:  column_template = '{} ±{}'       ; spaces = ['  ']
            matter = [column_template.format(*row) for row in column]

            # Header
            subcols = statkey, '1σ', '💀', '✔️'
            alignms = '>','<','>','>',
            spaces.append('') # no space after last subcol
            lens   = [len(s) for s in column[-1]]
            header = ''.join(['{:{}{}}{}'.format(*specs)
                for specs in zip(subcols,alignms,lens,spaces)])

            # Super-header
            if column_axis:
                width = len(header)
                if mn: width += 1 # coz ✔️ takes 2 chars
                super_header = column_axis + "=" + str(col_val)
                super_header = super_header.center(width,"_")
                header = super_header + "\n" + header

            # Append column to table
            matters = matters + [matter]
            headers = headers + [header]

        table = tabulate(matters, headers).replace('æ',' ')
        print(table)


##
import pytest

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


@functools.wraps(print_1d)
def _print_1d(*args,**kwargs):
    """Usage: As the usual print_1d, but makes it into a test.

    Features:
    - Enables re-use of ``old`` variable name (by capturing its value).
    - Parameterization -- pytest.mark.parametrize not used.
                          Avoids having to decorate an explicit function
                          (and thus enables naming functions through test_ind).
    - Capturing stdout -- The func print_1d() is only called once for each ``old``
                          (unlike a pytest fixture with capsys),
                          and thus it's fast.
    - splitlines() included.

    NB: the strip() functionality has been removed since the data is
        now written automatically (inluding bothersome trailing whitespaces)
        using the --replace option.
    """

    # Call actual print function. Capture stdout as ``new``.
    F = io.StringIO()
    with redirect_stdout(F):
        print_1d(*args,**kwargs)
    _new = F.getvalue().splitlines(True)

    if "--replace" in sys.argv:
        with open(__file__,"r") as F:
            orig_code = [ln for ln in F]
        def backtrack_until_finding(substr,lineno):
            while True:
                lineno -= 1
                if substr in orig_code[lineno]:
                    return lineno
        caller_lineno = inspect.currentframe().f_back.f_lineno
        nClose = backtrack_until_finding('"""\n', caller_lineno)
        nOpen  = backtrack_until_finding('"""\n', nClose)
        replacements.append(Replacement(_new,nOpen,nClose))

    elif "--print" in sys.argv:
        print("".join(_new))

    else:
        # Generate test functions
        global test_ind
        test_ind += 1

        # Capture ``old``
        _old = old.splitlines(True) # keepends

        # Loop over rows
        for lineno, (old_bound,new_bound) in enumerate(zip(_old,_new)):

            # Define test function.
            def compare(old_line=old_bound,new_line=new_bound):
                # assert old_line.strip() == new_line.strip()
                assert old_line == new_line

            # Register test
            lcls[f'test_{test_ind}_line_{lineno}'] = compare


##
__file__ = "tests/test_data.py"
savepath = save_dir(__file__)
xps = load_xps(savepath)
xps = ExperimentHypercube(xps)

xps_shorter = ExperimentHypercube([xp for xp in xps
    if getattr(xp,'da_method')!='LETKF'])

##
old = """
da_method=Climatology
Stats avrg'd wrt. ['seed'].
     |  ______N=None______
     |  rmse.a  1σ    💀✔️
---  -  ------------------
[0]  |   3.624 ±0.006 0 3

da_method=OptInterp
Stats avrg'd wrt. ['seed'].
     |  ______N=None______
     |  rmse.a  1σ    💀✔️
---  -  ------------------
[0]  |   0.941 ±0.001 0 3

da_method=EnKF
Stats avrg'd wrt. ['seed'].
     infl  |  _______N=10______  _______N=12______  _______N=14______
           |  rmse.a  1σ   💀✔️   rmse.a  1σ   💀✔️   rmse.a  1σ   💀✔️
---  ----  -  -----------------  -----------------  -----------------
[0]  1     |    4.55 ±0.09 0 3    4.33  ±0.07 0 3     4.19 ±0.08 0 3
[1]  1.01  |    4.4  ±0.04 0 3    4.24  ±0.1  0 3     3.96 ±0.1  0 3
[2]  1.02  |    4.36 ±0.06 0 3    4.08  ±0.2  0 3     3.76 ±0.1  0 3
[3]  1.04  |    4.16 ±0.1  0 3    3.88  ±0.07 0 3     2.88 ±0.3  0 3
[4]  1.07  |    3.92 ±0.2  0 3    3.6   ±0.2  0 3     2.7  ±0.3  0 3
[5]  1.1   |    3.84 ±0.1  0 3    3.462 ±0.03 0 3     2.28 ±0.3  0 3
[6]  1.2   |    3.58 ±0.1  0 3    2.92  ±0.05 0 3     1.28 ±0.2  0 3
[7]  1.4   |    3.43 ±0.06 0 3    2.52  ±0.1  0 3     0.92 ±0.2  0 3

da_method=LETKF
Stats avrg'd wrt. ['seed'].
      loc_rad  infl  |  _______N=10_______  _______N=12_______  _______N=14_______
                     |  rmse.a  1σ    💀✔️   rmse.a  1σ    💀✔️   rmse.a  1σ    💀✔️
----  -------  ----  -  ------------------  ------------------  ------------------
[0]       0.1  1     |  3.758  ±0.01  0 3   3.756  ±0.01  0 3   3.755  ±0.009 0 3
[1]       0.4  1     |  0.51   ±0.03  0 3   0.53   ±0.07  0 3   0.53   ±0.08  0 3
[2]       2    1     |  1.3    ±0.6   0 3   0.8    ±0.5   0 3   0.9    ±0.3   0 3
[3]       0.1  1.01  |  3.818  ±0.01  0 3   3.806  ±0.01  0 3   3.8    ±0.01  0 3
[4]       0.4  1.01  |  0.41   ±0.01  0 3   0.408  ±0.02  0 3   0.396  ±0.01  0 3
[5]       2    1.01  |  0.26   ±0.007 0 3   0.251  ±0.008 0 3   0.25   ±0.009 0 3
[6]       0.1  1.02  |  3.892  ±0.01  0 3   3.868  ±0.01  0 3   3.86   ±0.01  0 3
[7]       0.4  1.02  |  0.3752 ±0.004 0 3   0.375  ±0.005 0 3   0.3696 ±0.004 0 3
[8]       2    1.02  |  0.245  ±0.007 0 3   0.242  ±0.006 0 3   0.24   ±0.004 0 3
[9]       0.1  1.04  |  4.118  ±0.01  0 3   4.08   ±0.006 0 3   4.064  ±0.02  0 3
[10]      0.4  1.04  |  0.3616 ±0.004 0 3   0.36   ±0.003 0 3   0.3594 ±0.003 0 3
[11]      2    1.04  |  0.2454 ±0.003 0 3   0.2454 ±0.003 0 3   0.2448 ±0.003 0 3
[12]      0.1  1.07  |  nan    ±nan   3 0   nan    ±nan   3 0   nan    ±nan   3 0
[13]      0.4  1.07  |  0.3726 ±0.003 0 3   0.3732 ±0.003 0 3   0.3702 ±0.003 0 3
[14]      2    1.07  |  0.2664 ±0.002 0 3   0.2664 ±0.002 0 3   0.2664 ±0.002 0 3
[15]      0.1  1.1   |  nan    ±nan   3 0   nan    ±nan   3 0   nan    ±nan   3 0
[16]      0.4  1.1   |  0.3932 ±0.002 0 3   0.393  ±0.003 0 3   0.3924 ±0.002 0 3
[17]      2    1.1   |  0.2904 ±0.001 0 3   0.2916 ±0.001 0 3   0.2928 ±0.002 0 3
[18]      0.1  1.2   |  nan    ±nan   3 0   nan    ±nan   3 0   nan    ±nan   3 0
[19]      0.4  1.2   |  0.468  ±0.002 0 3   0.4692 ±0.002 0 3   0.4694 ±0.001 0 3
[20]      2    1.2   |  0.3716 ±0.002 0 3   0.3744 ±0.001 0 3   0.3762 ±0.001 0 3
[21]      0.1  1.4   |  nan    ±nan   3 0   nan    ±nan   3 0   nan    ±nan   3 0
[22]      0.4  1.4   |  0.5872 ±0.001 0 3   0.5884 ±0.002 0 3   0.5892 ±0.001 0 3
[23]      2    1.4   |  0.4978 ±0.001 0 3   0.5022 ±0.001 0 3   0.5052 ±0.001 0 3
"""
_print_1d(xps)

##
old = """
da_method=Climatology
Stats avrg'd wrt. ['seed'].
     |  _____N=None_____
     |  kurt.f  1σ  💀✔️
---  -  ----------------
[0]  |  nan    ±nan 3 0

da_method=OptInterp
Stats avrg'd wrt. ['seed'].
     |  _____N=None_____
     |  kurt.f  1σ  💀✔️
---  -  ----------------
[0]  |  nan    ±nan 3 0

da_method=EnKF
Stats avrg'd wrt. ['seed'].
     infl  |  ________N=10________  ________N=12________  ________N=14________
           |   kurt.f  1σ     💀✔️    kurt.f  1σ     💀✔️    kurt.f  1σ     💀✔️
---  ----  -  --------------------  --------------------  --------------------
[0]  1     |  -1.0138 ±0.001  0 3   -0.8604 ±0.003  0 3   -0.7544 ±0.002  0 3
[1]  1.01  |  -1.01   ±0.0007 0 3   -0.8658 ±0.001  0 3   -0.76   ±0.0006 0 3
[2]  1.02  |  -1.0128 ±0.002  0 3   -0.862  ±0.002  0 3   -0.758  ±0.002  0 3
[3]  1.04  |  -1.0122 ±0.001  0 3   -0.8669 ±0.0008 0 3   -0.758  ±0.001  0 3
[4]  1.07  |  -1.01   ±0.002  0 3   -0.8658 ±0.001  0 3   -0.7576 ±0.002  0 3
[5]  1.1   |  -1.013  ±0.0008 0 3   -0.8646 ±0.003  0 3   -0.7576 ±0.002  0 3
[6]  1.2   |  -1.008  ±0.003  0 3   -0.8704 ±0.004  0 3   -0.7576 ±0.004  0 3
[7]  1.4   |  -1.0104 ±0.004  0 3   -0.8628 ±0.003  0 3   -0.753  ±0.001  0 3

da_method=LETKF
Stats avrg'd wrt. ['seed'].
      loc_rad  infl  |  _________N=10________  _________N=12________  _________N=14________
                     |    kurt.f  1σ     💀✔️     kurt.f  1σ     💀✔️     kurt.f  1σ     💀✔️
----  -------  ----  -  ---------------------  ---------------------  ---------------------
[0]       0.1  1     |  -1.0162  ±0.001  0 3   -0.874   ±0.0007 0 3   -0.7679  ±0.0009 0 3
[1]       0.4  1     |  -1.008   ±0.001  0 3   -0.86    ±0.006  0 3   -0.75    ±0.003  0 3
[2]       2    1     |  -1.0128  ±0.002  0 3   -0.8664  ±0.003  0 3   -0.7552  ±0.004  0 3
[3]       0.1  1.01  |  -1.0158  ±0.001  0 3   -0.8684  ±0.001  0 3   -0.76266 ±0.0003 0 3
[4]       0.4  1.01  |  -1.0094  ±0.001  0 3   -0.8584  ±0.002  0 3   -0.7555  ±0.0006 0 3
[5]       2    1.01  |  -1.011   ±0.003  0 3   -0.8632  ±0.004  0 3   -0.7608  ±0.003  0 3
[6]       0.1  1.02  |  -1.0108  ±0.002  0 3   -0.8676  ±0.0003 0 3   -0.762   ±0.003  0 3
[7]       0.4  1.02  |  -1.0104  ±0.0005 0 3   -0.866   ±0.0004 0 3   -0.7542  ±0.003  0 3
[8]       2    1.02  |  -1.011   ±0.003  0 3   -0.864   ±0.003  0 3   -0.758   ±0.002  0 3
[9]       0.1  1.04  |  -1.012   ±0.001  0 3   -0.86472 ±0.0004 0 3   -0.7564  ±0.0002 0 3
[10]      0.4  1.04  |  -1.0084  ±0.002  0 3   -0.8592  ±0.003  0 3   -0.7518  ±0.003  0 3
[11]      2    1.04  |  -1.0086  ±0.001  0 3   -0.864   ±0.003  0 3   -0.7604  ±0.002  0 3
[12]      0.1  1.07  |  nan      ±nan    3 0   nan      ±nan    3 0   nan      ±nan    3 0
[13]      0.4  1.07  |  -1.0084  ±0.002  0 3   -0.8636  ±0.002  0 3   -0.7532  ±0.002  0 3
[14]      2    1.07  |  -1.0083  ±0.0007 0 3   -0.8699  ±0.0009 0 3   -0.7588  ±0.001  0 3
[15]      0.1  1.1   |  nan      ±nan    3 0   nan      ±nan    3 0   nan      ±nan    3 0
[16]      0.4  1.1   |  -1.0098  ±0.003  0 3   -0.8632  ±0.001  0 3   -0.7548  ±0.003  0 3
[17]      2    1.1   |  -1.0105  ±0.0006 0 3   -0.8652  ±0.001  0 3   -0.7576  ±0.002  0 3
[18]      0.1  1.2   |  nan      ±nan    3 0   nan      ±nan    3 0   nan      ±nan    3 0
[19]      0.4  1.2   |  -1.008   ±0.001  0 3   -0.8632  ±0.0006 0 3   -0.7536  ±0.001  0 3
[20]      2    1.2   |  -1.0109  ±0.0009 0 3   -0.8667  ±0.0007 0 3   -0.75864 ±0.0004 0 3
[21]      0.1  1.4   |  nan      ±nan    3 0   nan      ±nan    3 0   nan      ±nan    3 0
[22]      0.4  1.4   |  -1.00992 ±0.0004 0 3   -0.8632  ±0.002  0 3   -0.7566  ±0.003  0 3
[23]      2    1.4   |  -1.011   ±0.001  0 3   -0.868   ±0.002  0 3   -0.7596  ±0.001  0 3
"""
_print_1d(xps,statkey="kurt.f")

##
old = """
da_method=Climatology
Stats avrg'd wrt. ['seed'].
     |  ____infl=None_____
     |  rmse.a  1σ    💀✔️
---  -  ------------------
[0]  |   3.624 ±0.006 0 3

da_method=OptInterp
Stats avrg'd wrt. ['seed'].
     |  ____infl=None_____
     |  rmse.a  1σ    💀✔️
---  -  ------------------
[0]  |   0.941 ±0.001 0 3

da_method=EnKF
Stats avrg'd wrt. ['seed'].
      N  |  _____infl=1.0____  ____infl=1.01____  ____infl=1.02____  ____infl=1.04____  ___infl=1.07____  _____infl=1.1____  _____infl=1.2____  _____infl=1.4____
         |  rmse.a  1σ   💀✔️   rmse.a  1σ   💀✔️   rmse.a  1σ   💀✔️   rmse.a  1σ   💀✔️   rmse.a  1σ  💀✔️   rmse.a  1σ   💀✔️   rmse.a  1σ   💀✔️   rmse.a  1σ   💀✔️
---  --  -  -----------------  -----------------  -----------------  -----------------  ----------------  -----------------  -----------------  -----------------
[0]  10  |    4.55 ±0.09 0 3     4.4  ±0.04 0 3     4.36 ±0.06 0 3     4.16 ±0.1  0 3     3.92 ±0.2 0 3    3.84  ±0.1  0 3     3.58 ±0.1  0 3     3.43 ±0.06 0 3
[1]  12  |    4.33 ±0.07 0 3     4.24 ±0.1  0 3     4.08 ±0.2  0 3     3.88 ±0.07 0 3     3.6  ±0.2 0 3    3.462 ±0.03 0 3     2.92 ±0.05 0 3     2.52 ±0.1  0 3
[2]  14  |    4.19 ±0.08 0 3     3.96 ±0.1  0 3     3.76 ±0.1  0 3     2.88 ±0.3  0 3     2.7  ±0.3 0 3    2.28  ±0.3  0 3     1.28 ±0.2  0 3     0.92 ±0.2  0 3

da_method=LETKF
Stats avrg'd wrt. ['seed'].
      N  loc_rad  |  _____infl=1.0_____  ____infl=1.01_____  ____infl=1.02_____  ____infl=1.04_____  ____infl=1.07_____  _____infl=1.1_____  _____infl=1.2_____  _____infl=1.4_____
                  |  rmse.a  1σ    💀✔️   rmse.a  1σ    💀✔️   rmse.a  1σ    💀✔️   rmse.a  1σ    💀✔️   rmse.a  1σ    💀✔️   rmse.a  1σ    💀✔️   rmse.a  1σ    💀✔️   rmse.a  1σ    💀✔️
---  --  -------  -  ------------------  ------------------  ------------------  ------------------  ------------------  ------------------  ------------------  ------------------
[0]  10      0.1  |   3.758 ±0.01  0 3    3.818 ±0.01  0 3   3.892  ±0.01  0 3   4.118  ±0.01  0 3   nan    ±nan   3 0   nan    ±nan   3 0   nan    ±nan   3 0   nan    ±nan   3 0
[1]  10      0.4  |   0.51  ±0.03  0 3    0.41  ±0.01  0 3   0.3752 ±0.004 0 3   0.3616 ±0.004 0 3   0.3726 ±0.003 0 3   0.3932 ±0.002 0 3   0.468  ±0.002 0 3   0.5872 ±0.001 0 3
[2]  10      2    |   1.3   ±0.6   0 3    0.26  ±0.007 0 3   0.245  ±0.007 0 3   0.2454 ±0.003 0 3   0.2664 ±0.002 0 3   0.2904 ±0.001 0 3   0.3716 ±0.002 0 3   0.4978 ±0.001 0 3
[3]  12      0.1  |   3.756 ±0.01  0 3    3.806 ±0.01  0 3   3.868  ±0.01  0 3   4.08   ±0.006 0 3   nan    ±nan   3 0   nan    ±nan   3 0   nan    ±nan   3 0   nan    ±nan   3 0
[4]  12      0.4  |   0.53  ±0.07  0 3    0.408 ±0.02  0 3   0.375  ±0.005 0 3   0.36   ±0.003 0 3   0.3732 ±0.003 0 3   0.393  ±0.003 0 3   0.4692 ±0.002 0 3   0.5884 ±0.002 0 3
[5]  12      2    |   0.8   ±0.5   0 3    0.251 ±0.008 0 3   0.242  ±0.006 0 3   0.2454 ±0.003 0 3   0.2664 ±0.002 0 3   0.2916 ±0.001 0 3   0.3744 ±0.001 0 3   0.5022 ±0.001 0 3
[6]  14      0.1  |   3.755 ±0.009 0 3    3.8   ±0.01  0 3   3.86   ±0.01  0 3   4.064  ±0.02  0 3   nan    ±nan   3 0   nan    ±nan   3 0   nan    ±nan   3 0   nan    ±nan   3 0
[7]  14      0.4  |   0.53  ±0.08  0 3    0.396 ±0.01  0 3   0.3696 ±0.004 0 3   0.3594 ±0.003 0 3   0.3702 ±0.003 0 3   0.3924 ±0.002 0 3   0.4694 ±0.001 0 3   0.5892 ±0.001 0 3
[8]  14      2    |   0.9   ±0.3   0 3    0.25  ±0.009 0 3   0.24   ±0.004 0 3   0.2448 ±0.003 0 3   0.2664 ±0.002 0 3   0.2928 ±0.002 0 3   0.3762 ±0.001 0 3   0.5052 ±0.001 0 3
"""
_print_1d(xps,column_axis='infl')

##
old = """
N=None
Stats avrg'd wrt. ['seed'].
     da_method    |  ____infl=None_____
                  |  rmse.a  1σ    💀✔️
---  -----------  -  ------------------
[0]  Climatology  |   3.624 ±0.006 0 3
[1]  OptInterp    |   0.941 ±0.001 0 3

N=10
Stats avrg'd wrt. ['seed'].
     da_method  loc_rad  |  _____infl=1.0____  ____infl=1.01_____  ____infl=1.02_____  ____infl=1.04_____  ____infl=1.07_____  _____infl=1.1_____  _____infl=1.2_____  _____infl=1.4_____
                         |  rmse.a  1σ   💀✔️   rmse.a  1σ    💀✔️   rmse.a  1σ    💀✔️   rmse.a  1σ    💀✔️   rmse.a  1σ    💀✔️   rmse.a  1σ    💀✔️   rmse.a  1σ    💀✔️   rmse.a  1σ    💀✔️
---  ---------  -------  -  -----------------  ------------------  ------------------  ------------------  ------------------  ------------------  ------------------  ------------------
[0]  EnKF                |   4.55  ±0.09 0 3    4.4   ±0.04  0 3   4.36   ±0.06  0 3   4.16   ±0.1   0 3   3.92   ±0.2   0 3   3.84   ±0.1   0 3   3.58   ±0.1   0 3   3.43   ±0.06  0 3
[1]  LETKF          0.1  |   3.758 ±0.01 0 3    3.818 ±0.01  0 3   3.892  ±0.01  0 3   4.118  ±0.01  0 3   nan    ±nan   3 0   nan    ±nan   3 0   nan    ±nan   3 0   nan    ±nan   3 0
[2]  LETKF          0.4  |   0.51  ±0.03 0 3    0.41  ±0.01  0 3   0.3752 ±0.004 0 3   0.3616 ±0.004 0 3   0.3726 ±0.003 0 3   0.3932 ±0.002 0 3   0.468  ±0.002 0 3   0.5872 ±0.001 0 3
[3]  LETKF          2    |   1.3   ±0.6  0 3    0.26  ±0.007 0 3   0.245  ±0.007 0 3   0.2454 ±0.003 0 3   0.2664 ±0.002 0 3   0.2904 ±0.001 0 3   0.3716 ±0.002 0 3   0.4978 ±0.001 0 3

N=12
Stats avrg'd wrt. ['seed'].
     da_method  loc_rad  |  _____infl=1.0____  ____infl=1.01_____  ____infl=1.02_____  ____infl=1.04_____  ____infl=1.07_____  _____infl=1.1_____  _____infl=1.2_____  _____infl=1.4_____
                         |  rmse.a  1σ   💀✔️   rmse.a  1σ    💀✔️   rmse.a  1σ    💀✔️   rmse.a  1σ    💀✔️   rmse.a  1σ    💀✔️   rmse.a  1σ    💀✔️   rmse.a  1σ    💀✔️   rmse.a  1σ    💀✔️
---  ---------  -------  -  -----------------  ------------------  ------------------  ------------------  ------------------  ------------------  ------------------  ------------------
[0]  EnKF                |   4.33  ±0.07 0 3    4.24  ±0.1   0 3    4.08  ±0.2   0 3   3.88   ±0.07  0 3   3.6    ±0.2   0 3   3.462  ±0.03  0 3   2.92   ±0.05  0 3   2.52   ±0.1   0 3
[1]  LETKF          0.1  |   3.756 ±0.01 0 3    3.806 ±0.01  0 3    3.868 ±0.01  0 3   4.08   ±0.006 0 3   nan    ±nan   3 0   nan    ±nan   3 0   nan    ±nan   3 0   nan    ±nan   3 0
[2]  LETKF          0.4  |   0.53  ±0.07 0 3    0.408 ±0.02  0 3    0.375 ±0.005 0 3   0.36   ±0.003 0 3   0.3732 ±0.003 0 3   0.393  ±0.003 0 3   0.4692 ±0.002 0 3   0.5884 ±0.002 0 3
[3]  LETKF          2    |   0.8   ±0.5  0 3    0.251 ±0.008 0 3    0.242 ±0.006 0 3   0.2454 ±0.003 0 3   0.2664 ±0.002 0 3   0.2916 ±0.001 0 3   0.3744 ±0.001 0 3   0.5022 ±0.001 0 3

N=14
Stats avrg'd wrt. ['seed'].
     da_method  loc_rad  |  _____infl=1.0_____  ____infl=1.01_____  ____infl=1.02_____  ____infl=1.04_____  ____infl=1.07_____  _____infl=1.1_____  _____infl=1.2_____  _____infl=1.4_____
                         |  rmse.a  1σ    💀✔️   rmse.a  1σ    💀✔️   rmse.a  1σ    💀✔️   rmse.a  1σ    💀✔️   rmse.a  1σ    💀✔️   rmse.a  1σ    💀✔️   rmse.a  1σ    💀✔️   rmse.a  1σ    💀✔️
---  ---------  -------  -  ------------------  ------------------  ------------------  ------------------  ------------------  ------------------  ------------------  ------------------
[0]  EnKF                |   4.19  ±0.08  0 3    3.96  ±0.1   0 3   3.76   ±0.1   0 3   2.88   ±0.3   0 3   2.7    ±0.3   0 3   2.28   ±0.3   0 3   1.28   ±0.2   0 3   0.92   ±0.2   0 3
[1]  LETKF          0.1  |   3.755 ±0.009 0 3    3.8   ±0.01  0 3   3.86   ±0.01  0 3   4.064  ±0.02  0 3   nan    ±nan   3 0   nan    ±nan   3 0   nan    ±nan   3 0   nan    ±nan   3 0
[2]  LETKF          0.4  |   0.53  ±0.08  0 3    0.396 ±0.01  0 3   0.3696 ±0.004 0 3   0.3594 ±0.003 0 3   0.3702 ±0.003 0 3   0.3924 ±0.002 0 3   0.4694 ±0.001 0 3   0.5892 ±0.001 0 3
[3]  LETKF          2    |   0.9   ±0.3   0 3    0.25  ±0.009 0 3   0.24   ±0.004 0 3   0.2448 ±0.003 0 3   0.2664 ±0.002 0 3   0.2928 ±0.002 0 3   0.3762 ±0.001 0 3   0.5052 ±0.001 0 3
"""
_print_1d(xps,column_axis='infl',panel_axes=("N",))

##
# TODO:
# print_1d(xps,column_axis='da_method',panel_axes=("N",))
old = """
"""

##
old = """
da_method=Climatology
     seed  |  ___N=None___
           |  rmse.a  1σ
---  ----  -  ------------
[0]     2  |   3.632 ±0.02
[1]     3  |   3.612 ±0.02
[2]     4  |   3.628 ±0.02

da_method=OptInterp
     seed  |  ____N=None___
           |  rmse.a  1σ
---  ----  -  -------------
[0]     2  |  0.944  ±0.004
[1]     3  |  0.939  ±0.003
[2]     4  |  0.9396 ±0.003

da_method=EnKF
      infl  seed  |  ____N=10____  ____N=12____  ____N=14____
                  |  rmse.a  1σ    rmse.a  1σ    rmse.a  1σ
----  ----  ----  -  ------------  ------------  ------------
[0]   1        2  |    4.54 ±0.09    4.26 ±0.1     4.08 ±0.2 
[1]   1.01     2  |    4.48 ±0.08    4.41 ±0.09    4.04 ±0.2 
[2]   1.02     2  |    4.35 ±0.08    4.24 ±0.08    3.92 ±0.2 
[3]   1.04     2  |    4.4  ±0.07    3.94 ±0.09    3.4  ±0.2 
[4]   1.07     2  |    4    ±0.1     3.8  ±0.1     3.16 ±0.2 
[5]   1.1      2  |    3.9  ±0.1     3.44 ±0.2     2.76 ±0.3 
[6]   1.2      2  |    3.68 ±0.08    2.96 ±0.2     1.28 ±0.4 
[7]   1.4      2  |    3.5  ±0.1     2.34 ±0.3     1.16 ±0.2 
[8]   1        3  |    4.4  ±0.1     4.24 ±0.2     4.12 ±0.2 
[9]   1.01     3  |    4.34 ±0.1     4.04 ±0.2     3.68 ±0.2 
[10]  1.02     3  |    4.26 ±0.1     3.76 ±0.2     3.88 ±0.2 
[11]  1.04     3  |    3.92 ±0.2     3.74 ±0.1     2.5  ±0.6 
[12]  1.07     3  |    3.64 ±0.2     3.3  ±0.3     2.82 ±0.3 
[13]  1.1      3  |    3.64 ±0.2     3.44 ±0.2     2.34 ±0.3 
[14]  1.2      3  |    3.38 ±0.1     2.84 ±0.2     0.96 ±0.3 
[15]  1.4      3  |    3.32 ±0.1     2.48 ±0.2     0.55 ±0.05
[16]  1        4  |    4.72 ±0.07    4.47 ±0.07    4.36 ±0.1 
[17]  1.01     4  |    4.37 ±0.08    4.27 ±0.06    4.16 ±0.2 
[18]  1.02     4  |    4.45 ±0.07    4.28 ±0.07    3.5  ±0.5 
[19]  1.04     4  |    4.16 ±0.09    3.94 ±0.1     2.6  ±0.6 
[20]  1.07     4  |    4.16 ±0.09    3.72 ±0.1     2.2  ±0.6 
[21]  1.1      4  |    4.01 ±0.09    3.52 ±0.1     1.6  ±0.5 
[22]  1.2      4  |    3.7  ±0.1     2.96 ±0.2     1.62 ±0.3 
[23]  1.4      4  |    3.48 ±0.1     2.76 ±0.2     1.08 ±0.3 
"""
_print_1d(xps_shorter,mean_axes=None)

##
old = """
da_method=Climatology
Stats avrg'd wrt. [].
     seed  |  _____N=None_____
           |  rmse.a  1σ  💀✔️
---  ----  -  ----------------
[0]     2  |   3.632 ±nan 0 1
[1]     3  |   3.611 ±nan 0 1
[2]     4  |   3.628 ±nan 0 1

da_method=OptInterp
Stats avrg'd wrt. [].
     seed  |  _____N=None_____
           |  rmse.a  1σ  💀✔️
---  ----  -  ----------------
[0]     2  |  0.9436 ±nan 0 1
[1]     3  |  0.9393 ±nan 0 1
[2]     4  |  0.9398 ±nan 0 1

da_method=EnKF
Stats avrg'd wrt. [].
      infl  seed  |  ______N=10______  ______N=12______  ______N=14______
                  |  rmse.a  1σ  💀✔️   rmse.a  1σ  💀✔️   rmse.a  1σ  💀✔️
----  ----  ----  -  ----------------  ----------------  ----------------
[0]   1        2  |   4.531 ±nan 0 1    4.259 ±nan 0 1   4.098  ±nan 0 1
[1]   1.01     2  |   4.479 ±nan 0 1    4.406 ±nan 0 1   4.043  ±nan 0 1
[2]   1.02     2  |   4.351 ±nan 0 1    4.241 ±nan 0 1   3.911  ±nan 0 1
[3]   1.04     2  |   4.391 ±nan 0 1    3.945 ±nan 0 1   3.411  ±nan 0 1
[4]   1.07     2  |   3.997 ±nan 0 1    3.794 ±nan 0 1   3.157  ±nan 0 1
[5]   1.1      2  |   3.904 ±nan 0 1    3.429 ±nan 0 1   2.774  ±nan 0 1
[6]   1.2      2  |   3.68  ±nan 0 1    2.97  ±nan 0 1   1.273  ±nan 0 1
[7]   1.4      2  |   3.508 ±nan 0 1    2.335 ±nan 0 1   1.166  ±nan 0 1
[8]   1        3  |   4.393 ±nan 0 1    4.238 ±nan 0 1   4.119  ±nan 0 1
[9]   1.01     3  |   4.347 ±nan 0 1    4.041 ±nan 0 1   3.695  ±nan 0 1
[10]  1.02     3  |   4.254 ±nan 0 1    3.772 ±nan 0 1   3.884  ±nan 0 1
[11]  1.04     3  |   3.929 ±nan 0 1    3.734 ±nan 0 1   2.556  ±nan 0 1
[12]  1.07     3  |   3.626 ±nan 0 1    3.289 ±nan 0 1   2.839  ±nan 0 1
[13]  1.1      3  |   3.624 ±nan 0 1    3.444 ±nan 0 1   2.365  ±nan 0 1
[14]  1.2      3  |   3.377 ±nan 0 1    2.823 ±nan 0 1   0.9706 ±nan 0 1
[15]  1.4      3  |   3.316 ±nan 0 1    2.47  ±nan 0 1   0.5529 ±nan 0 1
[16]  1        4  |   4.712 ±nan 0 1    4.468 ±nan 0 1   4.356  ±nan 0 1
[17]  1.01     4  |   4.369 ±nan 0 1    4.266 ±nan 0 1   4.163  ±nan 0 1
[18]  1.02     4  |   4.454 ±nan 0 1    4.284 ±nan 0 1   3.49   ±nan 0 1
[19]  1.04     4  |   4.166 ±nan 0 1    3.939 ±nan 0 1   2.634  ±nan 0 1
[20]  1.07     4  |   4.166 ±nan 0 1    3.723 ±nan 0 1   2.162  ±nan 0 1
[21]  1.1      4  |   4.019 ±nan 0 1    3.511 ±nan 0 1   1.65   ±nan 0 1
[22]  1.2      4  |   3.707 ±nan 0 1    2.979 ±nan 0 1   1.6    ±nan 0 1
[23]  1.4      4  |   3.478 ±nan 0 1    2.776 ±nan 0 1   1.051  ±nan 0 1
"""
_print_1d(xps_shorter,mean_axes=())

##
old = """
da_method=Climatology
Stats avrg'd wrt. ['seed'].
     |  ______N=None______
     |  rmse.a  1σ    💀✔️
---  -  ------------------
[0]  |   3.624 ±0.006 0 3

da_method=OptInterp
Stats avrg'd wrt. ['seed'].
     |  ______N=None______
     |  rmse.a  1σ    💀✔️
---  -  ------------------
[0]  |   0.941 ±0.001 0 3

da_method=EnKF
Stats avrg'd wrt. ['seed', 'infl'].
     |  _______N=10______  ______N=12______  ______N=14______
     |  rmse.a  1σ   💀✔️   rmse.a  1σ  💀✔️   rmse.a  1σ  💀✔️
---  -  -----------------  ----------------  ----------------
[0]  |    4.03 ±0.08 0 24    3.64 ±0.1 0 24    2.76 ±0.2 0 24
"""
_print_1d(xps_shorter,mean_axes=("seed","infl"))

##
old = """
da_method=Climatology
Stats avrg'd wrt. ['seed'].
        rmse.a  1σ    💀✔️
---  -  -----------------
[0]  |   3.624 ±0.006 0 3

da_method=OptInterp
Stats avrg'd wrt. ['seed'].
        rmse.a  1σ    💀✔️
---  -  -----------------
[0]  |   0.941 ±0.001 0 3

da_method=EnKF
Stats avrg'd wrt. ['seed'].
       N  infl     rmse.a  1σ   💀✔️
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
_print_1d(xps_shorter,column_axis=None)

##
old = """
da_method=Climatology
     seed     rmse.a  1σ
---  ----  -  ------------
[0]     2  |   3.632 ±0.02
[1]     3  |   3.612 ±0.02
[2]     4  |   3.628 ±0.02

da_method=OptInterp
     seed     rmse.a  1σ
---  ----  -  -------------
[0]     2  |  0.944  ±0.004
[1]     3  |  0.939  ±0.003
[2]     4  |  0.9396 ±0.003

da_method=EnKF
       N  infl  seed     rmse.a  1σ
----  --  ----  ----  -  ------------
[0]   10  1        2  |    4.54 ±0.09
[1]   10  1.01     2  |    4.48 ±0.08
[2]   10  1.02     2  |    4.35 ±0.08
[3]   10  1.04     2  |    4.4  ±0.07
[4]   10  1.07     2  |    4    ±0.1 
[5]   10  1.1      2  |    3.9  ±0.1 
[6]   10  1.2      2  |    3.68 ±0.08
[7]   10  1.4      2  |    3.5  ±0.1 
[8]   12  1        2  |    4.26 ±0.1 
[9]   12  1.01     2  |    4.41 ±0.09
[10]  12  1.02     2  |    4.24 ±0.08
[11]  12  1.04     2  |    3.94 ±0.09
[12]  12  1.07     2  |    3.8  ±0.1 
[13]  12  1.1      2  |    3.44 ±0.2 
[14]  12  1.2      2  |    2.96 ±0.2 
[15]  12  1.4      2  |    2.34 ±0.3 
[16]  14  1        2  |    4.08 ±0.2 
[17]  14  1.01     2  |    4.04 ±0.2 
[18]  14  1.02     2  |    3.92 ±0.2 
[19]  14  1.04     2  |    3.4  ±0.2 
[20]  14  1.07     2  |    3.16 ±0.2 
[21]  14  1.1      2  |    2.76 ±0.3 
[22]  14  1.2      2  |    1.28 ±0.4 
[23]  14  1.4      2  |    1.16 ±0.2 
[24]  10  1        3  |    4.4  ±0.1 
[25]  10  1.01     3  |    4.34 ±0.1 
[26]  10  1.02     3  |    4.26 ±0.1 
[27]  10  1.04     3  |    3.92 ±0.2 
[28]  10  1.07     3  |    3.64 ±0.2 
[29]  10  1.1      3  |    3.64 ±0.2 
[30]  10  1.2      3  |    3.38 ±0.1 
[31]  10  1.4      3  |    3.32 ±0.1 
[32]  12  1        3  |    4.24 ±0.2 
[33]  12  1.01     3  |    4.04 ±0.2 
[34]  12  1.02     3  |    3.76 ±0.2 
[35]  12  1.04     3  |    3.74 ±0.1 
[36]  12  1.07     3  |    3.3  ±0.3 
[37]  12  1.1      3  |    3.44 ±0.2 
[38]  12  1.2      3  |    2.84 ±0.2 
[39]  12  1.4      3  |    2.48 ±0.2 
[40]  14  1        3  |    4.12 ±0.2 
[41]  14  1.01     3  |    3.68 ±0.2 
[42]  14  1.02     3  |    3.88 ±0.2 
[43]  14  1.04     3  |    2.5  ±0.6 
[44]  14  1.07     3  |    2.82 ±0.3 
[45]  14  1.1      3  |    2.34 ±0.3 
[46]  14  1.2      3  |    0.96 ±0.3 
[47]  14  1.4      3  |    0.55 ±0.05
[48]  10  1        4  |    4.72 ±0.07
[49]  10  1.01     4  |    4.37 ±0.08
[50]  10  1.02     4  |    4.45 ±0.07
[51]  10  1.04     4  |    4.16 ±0.09
[52]  10  1.07     4  |    4.16 ±0.09
[53]  10  1.1      4  |    4.01 ±0.09
[54]  10  1.2      4  |    3.7  ±0.1 
[55]  10  1.4      4  |    3.48 ±0.1 
[56]  12  1        4  |    4.47 ±0.07
[57]  12  1.01     4  |    4.27 ±0.06
[58]  12  1.02     4  |    4.28 ±0.07
[59]  12  1.04     4  |    3.94 ±0.1 
[60]  12  1.07     4  |    3.72 ±0.1 
[61]  12  1.1      4  |    3.52 ±0.1 
[62]  12  1.2      4  |    2.96 ±0.2 
[63]  12  1.4      4  |    2.76 ±0.2 
[64]  14  1        4  |    4.36 ±0.1 
[65]  14  1.01     4  |    4.16 ±0.2 
[66]  14  1.02     4  |    3.5  ±0.5 
[67]  14  1.04     4  |    2.6  ±0.6 
[68]  14  1.07     4  |    2.2  ±0.6 
[69]  14  1.1      4  |    1.6  ±0.5 
[70]  14  1.2      4  |    1.62 ±0.3 
[71]  14  1.4      4  |    1.08 ±0.3 
"""
_print_1d(xps_shorter,column_axis=None,mean_axes=None)

##

if "--replace" in sys.argv:
    with open(__file__,"r") as F:
        orig_code = [ln for ln in F]

    new_code = orig_code[0:replacements[0].nOpen+1]

    for i,replacement in enumerate(replacements):

        new_code += replacement.lines[1:]

        try:
            nEnd = replacements[i+1].nOpen+1
        except IndexError:
            nEnd = len(orig_code)
        new_code += orig_code[replacement.nClose : nEnd]

    # Don't overwrite! This allows for diffing.
    with open(__file__+".new", "w") as F:
        for line in new_code:
            # F.write(line.rstrip()+"\n")
            F.write(line)


##
