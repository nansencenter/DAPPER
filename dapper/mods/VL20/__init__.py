"""
Single-scale Lorenz-96 with an added thermodynamic component.

Refs: `bib.vissio2020mechanics`
"""
import numpy as np


class model_instance():
    """
    Use OOP to facilitate having multiple parameter settings simultaneously.
    """
    def __init__(self, nX=18, F=10, G=10, alpha=1, gamma=1):
        # System size
        self.nX = nX       # num of gridpoints
        self.M = 2*nX     # => Total state length

        # Other parameters
        self.F = F          # forcing for X
        self.G = G          # forcing for theta
        self.alpha = alpha  # energy conversion coefficient
        self.gamma = gamma  # dissipation coefficient

        # Init with perturbation
        self.x0 = np.eye(self.M)[0]

    def shift(self, x, n):
        return np.roll(x, -n, axis=-1)

    def unpack(self, x):
        """Unpack model input to model parameters and state vector"""
        X = x[..., :self.nX]
        theta = x[..., self.nX:]
        return self.nX, self.F, self.G, self.alpha, self.gamma, X, theta

    def dxdt(self, x):
        """Full (coupled) dxdt."""
        nX, F, G, alpha, gamma, X, theta = self.unpack(x)
        d = np.zeros_like(x)
        d[..., :nX] = (self.shift(X, 1)-self.shift(X, -2))*self.shift(X, -1)
        d[..., :nX] -= gamma*self.shift(X, 0)
        d[..., :nX] += -alpha*self.shift(theta, 0) + F
        d[..., nX:] = self.shift(X, 1)*self.shift(theta, 2) \
            - self.shift(X, -1)*self.shift(theta, -2)
        d[..., nX:] -= gamma*self.shift(theta, 0)
        d[..., nX:] += alpha*self.shift(X, 0) + G

        return d

    def d2x_dtdx(self, x):
        """Tangent linear model"""
        nX, F, G, alpha, gamma, X, theta = self.unpack(x)
        def md(i): return np.mod(i, nX)  # modulo

        F = np.eye(self.M)
        F = F*(-gamma)
        for k in range(nX):
            # dX/dX
            F[k, md(k-1)] = X[md(k+1)]-X[k-2]
            F[k, md(k+1)] = X[k-1]
            F[k, md(k-2)] = -X[k-1]
            # dX/dtheta
            F[k, k + nX] = -alpha
            # dtheta/dX
            F[k + nX, k+1] = theta[md(k+2)]
            F[k + nX, md(k-1)] = -theta[k-2]
            F[k + nX, k] = alpha
            # dtheta/dtheta
            F[k + nX, md(k+2) + nX] = X[md(k+1)]
            F[k + nX, md(k-2) + nX] = -X[k-1]

        return F
