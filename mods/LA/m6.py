
# TODO: SEE MATLAB m6 FOR REST

from mods.LA.even2009 import *


F.tlm   = @(t,dt,x) damp*Fmat;
F.mod   = @(t,dt,x) damp*Fmat * x;
