% File:           mexcmd.m
%
% Created:        31/08/2007
%
% Last modified:  08/02/2008
%
% Author:         Pavel Sakov
%                 CSIRO Marine and Atmospheric Research
%                 NERSC
%
% Purpose:        Contains command line for compiling Fortran code for the QG
%                 model by Matlab's MEX compiler.
%
% Description:    
%
% Revisions:      

%% Copyright (C) 2008 Pavel Sakov
%% 
%% This file is part of EnKF-Matlab. EnKF-Matlab is a free software. See 
%% LICENSE for details.

mex -fortran -o QG_step_f utils.f90 parameters.f90 helmholtz.f90 calc.f90 qgflux.f90 qgstep.f90  mexf90.f90 qgstep_mex.f90
