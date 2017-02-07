% function [] = qgplay(psi, q, t, delay, dn)
%
% Plays PSI and Q fields read by qgread() as a movie.
%
% @param psi - psi (stream function); input; use output from qgread()
% @param q - q (vorticity); input; use output from qgread()
% @param t - t (time); input; use output from qgread()
% @param dt - time delay between frames; input
% @param dn -- stop every dn dumps; input

% File:           qgplay.m
%
% Created:        15/08/2007
%
% Last modified:  08/02/2008
%
% Author:         Pavel Sakov
%
% Purpose:        Plays PSI and Q fields read by qgread() as a movie.
%
% Description:
%
% Revisions:

%% Copyright (C) 2008 Pavel Sakov
%% 
%% This file is part of EnKF-Matlab. EnKF-Matlab is a free software. See 
%% LICENSE for details.

function [] = qgplay(psi, q, t, delay, dn)
    figure
    [nr, ny, nx] = size(psi);
  
    psi_max = max(max(max(abs(psi))));
    psi_range = [-psi_max psi_max];

    q_max = max(max(max(abs(q))));
    q_range = [-q_max q_max];
  
    for dump = 1 : nr
        subplot(1, 2, 1);
        A = squeeze(psi(dump, :, :));
        imagesc(A, psi_range);
        str = sprintf('\\Psi, dump = %d, t = %.0f', dump, t(dump));
        title(str);
        axis image;
    
        subplot(1, 2, 2);
        B = squeeze(q(dump, :, :));
        imagesc(B, q_range);
        title('Q');
        axis image;
    
        pause(delay);

        if exist('dn', 'var') & mod(dump, dn) == 0
            pause
        end
    end
  
    return
