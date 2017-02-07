% function [psi, q, psimean, qmean, t] = qgread(fname)
%
% Reads PSI and Q fields from the QG model output file. Also reads time array
% for the dumps in the output file and calculates mean fields.
%
% @param fname - output file name
% @return psi - 3D array with PSI fields (nr, 129, 129)
% @return q - 3D array with Q fields (nr, 129, 129)
% @return psimean - mean PSI field (129, 129)
% @return qmean - mean Q field (129, 129)
% @return t - array with time values for the dumps in the output file

% File:           qread.m
%
% Created:        31/08/2007
%
% Last modified:  08/02/2008
%
% Author:         Pavel Sakov
%                 CSIRO Marine and Atmospheric Research
%                 NERSC
% Purpose:        Reads PSI and Q fields from the QG model output file. Also
%                 reads time array for the dumps in the output file and
%                 calculates mean fields.
%
% Description:
%
% Revisions:

%% Copyright (C) 2008 Pavel Sakov
%% 
%% This file is part of EnKF-Matlab. EnKF-Matlab is a free software. See 
%% LICENSE for details.

function [psi, q, psimean, qmean, t] = qgread(fname)

    % OLD
    %nc = netcdf(fname, 'nowrite');
    %psi = nc{'psi'}(:, :, :);
    %q = nc{'q'}(:, :, :);
    %t = nc{'t'}(:);

	% NEW
	% ncdisp(fname)
	ncid = netcdf.open(fname);
	psi  = permute( netcdf.getVar(ncid, netcdf.inqVarID(ncid,'psi')), [3 2 1]);
	q    = permute( netcdf.getVar(ncid, netcdf.inqVarID(ncid,'q'))  , [3 2 1]);
	t    = netcdf.getVar(ncid, netcdf.inqVarID(ncid,'t'));

    psimean = squeeze(mean(psi));
    qmean = squeeze(mean(q));

  return
