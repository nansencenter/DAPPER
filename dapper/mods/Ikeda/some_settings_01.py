"""Settings that produce somewhat interesting/challenging DA problems."""

import numpy as np
import dapper.mods as modelling

from dapper.mods.Ikeda import step, x0, Tplot, LPs

tseq = modelling.Chronology(1, dko=1, Ko=1000, Tplot=Tplot, BurnIn=4*Tplot)

Nx = len(x0)

Dyn = {
    'M': Nx,
    'model': step,
    'noise': 0,
}

X0 = modelling.GaussRV(C=.1, mu=x0)

jj = np.arange(Nx)  # obs_inds
Obs = modelling.partial_Id_Obs(Nx, jj)
Obs['noise'] = .1  # modelling.GaussRV(C=CovMat(1*eye(Nx)))

HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)

HMM.liveplotters = LPs(jj)


####################
# Suggested tuning
####################
