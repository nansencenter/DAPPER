#Just using bocquet2019 settings for format and example; reproduce paper results if found

from dapper.mods.NS2D import Model, LP_setup
import dapper.mods as modelling
from dapper.mods.Lorenz96 import LPs
from dapper.tools.localization import nd_Id_localization
import numpy as np

noise_amp = 0.005
System = Model(T=1, N=64, dt=0.0001, nu = 0.01)
Nx = System.Nx

tseq = modelling.Chronology(System.dt, dko=1 , BurnIn=0.1, T=1)

Dyn = {
    "M" : np.prod((Nx, Nx)),
    "model": System.step,
    "linear": System.dstep_dx,
    "noise": 0,
}
X0 = modelling.RV(
    M=Dyn["M"],
    func=lambda N: System.x0.flatten()[None, :] + noise_amp * np.random.randn(N, Dyn["M"]) #Initial perturbation is noise_amp * N(0, 1) for each gridpoint
)

#X0 = lambda N: np.tile(System.x0.flatten() + 0 * np.random.randn(N, Dyn["M"]), (N, 1))
Obs = modelling.Id_Obs(Nx**2)
Obs["noise"] = 1

Obs["localizer"] = nd_Id_localization((Nx,), (4, ))

rstream = np.random.RandomState()
jj = modelling.linspace_int(Nx, Nx)
max_offset = jj[1] - jj[0]
def obs_inds(ko):
    def random_offset():
        rstream.seed(ko)
        u = rstream.rand()
        return int(np.floor(max_offset * u))
    return jj + random_offset()

def obs_now(ko):
    jj = obs_inds(ko)
    shape = (Nx, Nx)
    @modelling.ens_compatible
    def hmod(E):
        return E[jj]
    # Localization.
    batch_shape = [4, 4]  # width (in grid points) of each state batch.
    # Increasing the width
    #  => quicker analysis (but less rel. speed-up by parallelzt., depending on NPROC)
    #  => worse (increased) rmse (but width 4 is only slightly worse than 1);
    #     if inflation is applied locally, then rmse might actually improve.
    localizer = nd_Id_localization((shape)[::-1], batch_shape[::-1], jj, periodic=False)

    Obs = {
        "M": Nx,
        "model": hmod,
        "noise": modelling.GaussRV(C=4 * np.eye(Nx)),
        "localizer": localizer,
    }

    # Moving localization mask for smoothers:
    Obs["loc_shift"] = lambda ii, dt: ii  # no movement (suboptimal, but easy)

    # Jacobian left unspecified coz it's (usually) employed by methods that
    # compute full cov, which in this case is too big.

    return modelling.Operator(**Obs)


Obs = dict(time_dependent=lambda ko: obs_now(ko))
HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0, LP=LP_setup(obs_inds))
# HMM.liveplotters = LP_setup(obs_inds)