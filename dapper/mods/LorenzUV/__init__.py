"""The 2-scale/layer/speed coupled version of Lorenz-96.

See `bib.wilks2005effects`
- U:  large amp, low frequency vars: convective events
- V:  small amp, high frequency vars: large-scale synoptic events

Typically, the DA system will only use the truncated system
(containing only the U variables),
where the V's are parameterized as model noise,
while the truth is simulated by the full system.

Stochastic parmateterization:
Wilks: benefit of including stochastic noise negligible
unless its temporal auto-corr is taken into account (as AR(1))
(but spatial auto-corr can be neglected).
But AR(1) noise is technically difficult because DAPPER
is built around the Markov assumption.
"""
import numpy as np

import dapper.mods.Lorenz96 as L96
import dapper.tools.liveplotting as LP
from dapper.mods.integration import integrate_TLM


def reversible(fun):
    """Reverse input/output (instead of manipulating indices)."""
    def newfun(x, *args, reverse=False, **kwargs):
        if reverse:
            x = np.flip(x)  # flip ALL dims
        y = fun(x, *args, **kwargs)  # call fun()
        if reverse:
            y = np.flip(y)  # flip ALL dims
        return y
    return newfun


dxdt_auto     = reversible(L96.dxdt_autonomous)
d2x_dtdx_auto = reversible(L96.d2x_dtdx)


class model_instance():
    """Use OOP to facilitate having multiple parameter settings simultaneously.

    Default parameters from `bib.wilks2005effects`.
    """

    def __init__(self, nU=8, J=32, F=20, h=1, b=10, c=10):

        # System size
        self.nU = nU        # num of U
        self.J  = J         # num of V per U
        self.M  = (J+1)*nU  # => Total state length

        # Other parameters
        self.F  = F  # forcing
        self.h  = h  # coupling constant
        self.b  = b  # Spatial scale ratio
        self.c  = c  # time scale ratio

        # Indices for coupling
        self.iiU = (np.arange(J*nU)/J).astype(int)
        self.iiV = np.arange(J*nU)

        # Init with perturbation
        self.x0 = np.eye(self.M)[0]

    def unpack(self, x):
        nU, J, h, b, c = self.nU, self.J, self.h, self.b, self.c
        U, V = x[..., :nU], x[..., nU:]
        return nU, J, h, b, c, U, V

    def dxdt_trunc(self, x):
        """Compute truncated `dxdt:` slow variables (`U`) only."""
        assert x.shape[-1] == self.nU
        return dxdt_auto(x) + self.F

    def dxdt_parameterized(self, x, t):
        """Compute truncated `dxdt` with parameterization of fast variables (`V`)."""
        d  = self.dxdt_trunc(x)
        d -= self.prmzt(x, t)  # must (of course) be set first
        return d

    def dxdt(self, x):
        """Compute full (coupled) `dxdt`."""
        nU, J, h, b, c, U, V = self.unpack(x)

        d = np.zeros_like(x)
        d[..., :nU]  = self.dxdt_trunc(U)                                # dU/dt
        d[..., :nU] += -h*c/b * V.reshape(V.shape[:-1]+(nU, J)).sum(-1)  # Couple U<--V
        d[..., nU:]  = c/b*dxdt_auto(b*V, reverse=True)  # dV/dt
        d[..., nU:] += h*c/b * U[..., self.iiU]          # Couple V<--U

        return d

    def d2x_dtdx(self, x):
        nU, J, h, b, c, U, V = self.unpack(x)
        iiU, iiV = self.iiU, self.iiV+nU

        F = np.zeros((self.M, self.M))
        F[:nU, :nU] = d2x_dtdx_auto(U)  # dU/dU
        F[iiU, iiV] = -h*c/b            # dU/dV
        F[nU:, nU:] = c * d2x_dtdx_auto(b*V, reverse=True)  # dV/dV
        F[iiV, iiU] = h*c/b                                 # dV/dU

        return F

    def dstep_dx(self, x, t, dt):
        return integrate_TLM(self.d2x_dtdx_auto(x), dt, method='analytic')

    def LPs(self, jj):
        return [(1, LP.spatial1d(jj, dims=list(range(self.nU))))]
