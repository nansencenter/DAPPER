% function [] = qgplot(psi, q)
%
% Plots single dumps of PSI and Q fields, e.g. from arrays read by qgread().
%
% @param psi - PSI field
% @param q - Q field

% File:           qread.m
%
% Created:        31/08/2007
%
% Last modified:  08/02/2008
%
% Author:         Pavel Sakov
%                 CSIRO Marine and Atmospheric Research
%                 NERSC
% Purpose:        Plots single dumps of PSI and Q fields, e.g. from arrays
%                 read by qgread().
%
% Description:
%
% Revisions:

%% Copyright (C) 2008 Pavel Sakov
%% 
%% This file is part of EnKF-Matlab. EnKF-Matlab is a free software. See 
%% LICENSE for details.

function [] = qgplot(psi, q)
    
    size_psi = size(psi);
    
    if length(size(psi)) > 2
        psi = squeeze(psi);
        if length(size(psi)) > 2
            error(sprintf('\n  error: qgplot(): size(psi) = %d > 2\n', length(size(psi))));
        end
    elseif size(psi, 2) == 1
        nx = sqrt(size(psi, 1));
        if nx ~= floor(nx)
            error(sprintf('\n  error: qgplot(): size(psi) = %d x %d; sqrt(%d) = %.2f... confused...\n', size(psi, 1), size(psi, 2), sqrt(size(psi, 1))));
        end
        psi = reshape(psi, nx, nx);
    end
    if length(size(q)) > 2
        q = squeeze(q);
        if length(size(q)) > 2
            error(sprintf('\n  error: qgplot(): size(q) = %d > 2\n', length(size(q))));
        end
    elseif size(q, 2) == 1
        nx = sqrt(size(q, 1));
        if nx ~= floor(nx)
            error(sprintf('\n  error: qgplot(): size(q) = %d x %d; sqrt(%d) = %.2f... confused...\n', size(q, 1), size(q, 2), sqrt(size(q, 1))));
        end
        q = reshape(q, nx, nx);
    end
    
    psi_range = range(psi);
    psi_max = max(abs(psi_range));
    psi_range = [-psi_max psi_max];

    q_range = range(q);
    q_max = max(abs(q_range));
    q_range = [-q_max q_max];

    figure
    subplot(1, 2, 1);
    imagesc(psi, psi_range);
    title('\Psi');
    axis image;
    subplot(1, 2, 2);
    imagesc(q, q_range);
    title('Q');
    axis image;
  
    return
