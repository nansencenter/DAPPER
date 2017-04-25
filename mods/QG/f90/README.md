
Set pwd:

    $ cd DAPPER/mods/QG/f90

### This `f90` directory
contains Fortran90 code for the QG model for building
 1. a python extension module
 2. a standalone program
Both requires an f90 compiler (tested with g95, gfortran).
In addition (1) requires `f2py`, while (2) requires `netcdf` libraries.

For DAPPER, only the python module `py_mod` is required. To build it, run:

    $ f2py -c utils.f90 parameters.f90 helmholtz.f90 calc.f90 qgflux.f90 qgstep.f90 interface.f90 -m py_mod

If you wish, you may build the standalone executable `qg` as well. To do so, run

    $ make qg

after having adapted the `Makefile` to your system.

TODO: Download or gen samples?


### Details for compiling the standalone:
The following was done to make QG compile on my Mac
- get netcdf:
    brew install netcdf --with-fortran
- In makefile, changed to: 
		FC = gfortran-5
		NCLIB = /usr/local/lib/libnetcdff.dylib /usr/local/lib/libnetcdf.dylib
- Matlab has a new netcdf interface. Therefore, qgread.m should use
  	% ncdisp(fname)
  	ncid = netcdf.open(fname);
  	psi  = permute( netcdf.getVar(ncid, netcdf.inqVarID(ncid,'psi')), [3 2 1]);
  	q    = permute( netcdf.getVar(ncid, netcdf.inqVarID(ncid,'q'))  , [3 2 1]);
  	t    = netcdf.getVar(ncid, netcdf.inqVarID(ncid,'t'));


