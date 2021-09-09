"""Uses `nU`, `J`, `F` as in `dapper.mods.LorenzUV` ie. from `bib.wilks2005effects`.

Obs settings taken from different places (=> quasi-linear regime).
"""

import numpy as np

import dapper.mods as modelling
from dapper.mods.LorenzUV import model_instance

from ..utils import rel2mods

LUV = model_instance()
nU = LUV.nU

# Wilks2005 uses dt=1e-4 with RK4 for the full model,
# and dt=5e-3 with RK2 for the forecast/truncated model.
# As berry2014linear notes, this is possible coz
# "numerical stiffness disappears when fast processes are removed".

################
# Full
################

# tseq = modelling.Chronology(dt=0.001,dto=0.05,T=4**3,BurnIn=6) # allows using rk2
tseq = modelling.Chronology(dt=0.005, dto=0.05, T=4**3, BurnIn=6)  # requires rk4


Dyn = {
    'M': LUV.M,
    'model': modelling.with_rk4(LUV.dxdt, autonom=True),
    'noise': 0,
    'linear': LUV.dstep_dx,
}

X0 = modelling.GaussRV(mu=LUV.x0, C=0.01)

R = 0.1
jj = np.arange(nU)
Obs = modelling.partial_Id_Obs(LUV.M, jj)
Obs['noise'] = R

other = {'name': rel2mods(__file__)+'_full'}
HMM_full = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0, LP=LUV.LPs(jj), **other)


################
# Truncated
################

# Just change dt from 005 to 05
tseq = modelling.Chronology(dt=0.05, dto=0.05, T=4**3, BurnIn=6)

Dyn = {
    'M': nU,
    'model': modelling.with_rk4(LUV.dxdt_parameterized),
    'noise': 0,
}

X0 = modelling.GaussRV(mu=LUV.x0[:nU], C=0.01)

jj = np.arange(nU)
Obs = modelling.partial_Id_Obs(nU, jj)
Obs['noise'] = R

other = {'name': rel2mods(__file__)+'_trunc'}
HMM_trunc = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0, LP=LUV.LPs(jj), **other)

LUV.prmzt = lambda x, t: polynom_prmzt(x, t, 1)


def polynom_prmzt(x, t, order):
    """
    Polynomial (deterministic) parameterization of fast variables (Y).

    NB: Only valid for system settings of Wilks'2005.

    Note: In order to observe an improvement in DA performance w
          higher orders, the EnKF must be reasonably tuned with
          There is very little improvement gained above order=1.
    """
    if order == 4:
        # From Wilks
        d = 0.262 + 1.45*x - 0.0121*x**2 - 0.00713*x**3 + 0.000296*x**4
    elif order == 3:
        # From Arnold
        d = 0.341 + 1.30*x - 0.0136*x**2 - 0.00235*x**3
    elif order == 1:
        # From me -- see AdInf/illust_parameterizations.py
        d = 0.74 + 0.82*x
    elif order == 0:
        # From me -- see AdInf/illust_parameterizations.py
        d = 3.82
    elif order == -1:
        # Leave as dxdt_trunc
        d = 0
    else:
        raise NotImplementedError
    return d


####################
# Suggested tuning
####################
# Using HMM_full                    # Expected rmse.a:
# xps += Climatology()              # 0.93
# xps += Var3D(xB=2.0)              # 0.39
# xps += EnKF_N(N=20)               # 0.27
