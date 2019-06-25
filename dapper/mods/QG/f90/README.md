This model was taken from Sakov's EnKF-Matlab package (changelog at below). Licence reproduced below.

        Copyright (C) 2008, 2009 Pavel Sakov

        Redistribution and use of material from the package EnKF-Matlab, with or
        without modification, are permitted provided that the following conditions are 
        met:
        
           1. Redistributions of material must retain the above copyright notice, this
              list of conditions and the following disclaimer.
           2. The name of the author may not be used to endorse or promote products
              derived from this software without specific prior written permission.
        
        THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR IMPLIED 
        WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
        MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
        EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
        EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
        OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
        INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
        CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
        IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
        OF SUCH DAMAGE.

Compilate as follows...

### This `f90` directory
contains Fortran90 code for the QG model for building

 1. a python extension module `py_mod`
 2. a standalone program 

Both requires an f90 compiler (tested with g95 and gfortran).  
In addition  `py_mod` requires `f2py`, while `qg` requires `netcdf` libraries.  
DAPPER only requires `py_mod` (also to generate intial sample).


### For DAPPER,
To build `py_mod`, run:

    $ cd dapper/mods/QG/f90
    $ rm -rf py_mod.cpython-* __pycache__
    $ f2py -c utils.f90 parameters.f90 helmholtz.f90 calc.f90 qgflux.f90 qgstep.f90 interface.f90 -m py_mod



### For the standalone executable `qg`
(not required for DAPPER), adapted the `Makefile` to your system, and run

    $ make qg

Example: here's how I compiled the standalone on my Mac:
- get netcdf: `brew install netcdf --with-fortran`
- In makefile, changed to: 
      FC = gfortran-5
      NCLIB = /usr/local/lib/libnetcdff.dylib /usr/local/lib/libnetcdf.dylib
- Matlab has a new netcdf interface. Therefore, qgread.m should use
      % ncdisp(fname)
      ncid = netcdf.open(fname);
      psi  = permute( netcdf.getVar(ncid, netcdf.inqVarID(ncid,'psi')), [3 2 1]);
      q    = permute( netcdf.getVar(ncid, netcdf.inqVarID(ncid,'q'))  , [3 2 1]);
      t    = netcdf.getVar(ncid, netcdf.inqVarID(ncid,'t'));





### Changelog since Sakov's enkf-matlab:
 - Rm Matlab interface funcs: qgplay.m qgplot.m qgread.m mexcmd.m qgstep_mex.f90 mexf90.f90 
 - Modified makefile, as described above.
 - In parameters.f90:
   * Capitalized m,n
   * Swapped NY1 and NX1 in definitions of M,N 
 - Made `interface.f90` for Python.

