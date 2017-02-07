## Copyright (C) 2008 Pavel Sakov
## 
## This file is part of EnKF-Matlab. EnKF-Matlab is a free software. See 
## LICENSE for details.

SHELL = /bin/bash

FC = gfortran-5
FCFLAGS = -g -O2 -Wall
LDFLAGS = $(FCFLAGS)
LD = $(FC)

INCDIR = /usr/local/lib
NCINC = -I/usr/local/include -I$(INCDIR)
NCLIB = /usr/local/lib/libnetcdff.dylib /usr/local/lib/libnetcdf.dylib

QGSRC =\
utils.f90\
nfw.f90\
calc.f90\
parameters.f90\
data.f90\
helmholtz.f90\
qgflux.f90\
qgstep.f90\
qg.f90

MEXSRC =\
mexf90.f90\
qgstep_mex.f90

QGOBJ = $(QGSRC:.f90=.o)

MEXOBJ = $(MEXSRC:.f90=.o)

ROOT = $(shell pwd)

INC = -I$(ROOT)/mod $(NCINC)

all:: qg

mod:
	@rm -f *.mod
	@if [ ! -d "mod" ] ; then mkdir "mod" ; fi
	@echo "WD = $(PWD)"

qg: mod $(QGOBJ)
	$(LD) $(LDFLAGS) -o qg $(QGOBJ) $(NCLIB)

clean:
	rm -rf qg mod *.mod *.o *.log *.mexa64

%.o: %.f90
	$(FC) $(FCFLAGS) $(INC) -c $< -o $@
	$(shell mv -f *.mod mod 2>/dev/null)
