# QG Fortran model

### From Sakov's `enkf-matlab`

This model was taken from Sakov's EnKF-Matlab package.
Licence reproduced in THIRD_PARTY_NOTICES.

Changelog since `enkf-matlab`

- Rm Matlab interface functions: `qgplay.m qgplot.m qgread.m mexcmd.m qgstep_mex.f90 mexf90.f90`
- Modified makefile, as described above.
- In parameters.f90:
  - Capitalized m,n
  - Swapped `NY1` and `NX1` in definitions of M,N
  - Changed typing of arrays in `interface.f90`, `qgflux.f90`, `qgstep.f90` to allocatable.
- Made `interface.f90` for Python.

### This `f90` directory

...contains Fortran-90 code for the QG model for building

1. A python extension module `py_mod`
2. A standalone program

DAPPER only requires `py_mod` (also to generate initial sample).
Both require an `f90` compiler (tested with `g95` and `gfortran`).  
In addition  `py_mod` requires `f2py`, while `qg` requires `netcdf` libraries.


### For DAPPER

`py_mod` requires `gcc`, `gfortran`, and `f2py` (bundled with `numpy`).

**Install compilers.** Use one of:

```bash
brew install gcc gfortran # macOS
apt install gcc gfortran # Ubuntu
conda install -c conda-forge gcc gfortran # general
```

**Build.** From the repo root:

```bash
cd dapper/mods/QG/f90
rm -rf py_mod.cpython-* __pycache__  # remove stale build artifacts before rebuilding
export FC=gfortran-mp-14 # indicate the appropriate gfortran
python -m numpy.f2py -c utils.f90 parameters.f90 helmholtz.f90 calc.f90 qgflux.f90 qgstep.f90 interface.f90 -m py_mod
```

### For the standalone executable `qg`

(not required for DAPPER), adapt the `Makefile` to your system, and run

    make qg

Example: here's how I compiled the standalone on my Mac:

- Get `netcdf`
- In makefile, changed to:

      FC = gfortran-5
      NCLIB = /usr/local/lib/libnetcdff.dylib /usr/local/lib/libnetcdf.dylib

- Matlab has a new `netcdf` interface. Therefore, `qgread.m` should use

      % ncdisp(fname)
      ncid = netcdf.open(fname);
      psi  = permute( netcdf.getVar(ncid, netcdf.inqVarID(ncid,'psi')), [3 2 1]);
      q    = permute( netcdf.getVar(ncid, netcdf.inqVarID(ncid,'q'))  , [3 2 1]);
      t    = netcdf.getVar(ncid, netcdf.inqVarID(ncid,'t'));
