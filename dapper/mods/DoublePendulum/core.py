from dapper import *

# Based on double_pendulum_animated.html from matplotlib.org.
# which was based on physics.usyd.edu.au/~wheat/dpend_html/solve_dpend.c

G  = 9.8  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 1.0  # mass of pendulum 2 in kg

# Initial condition: th1, w1, th2, w2
x0 = np.radians([ 130,  0, -10,  0])

@ens_compatible
def dxdt(x):
    th1, w1, th2, w2 = x

    dydx = np.zeros_like(x)

    dydx[0] = w1  # d(th1)/dt =: w1
    dydx[2] = w2  # d(th1)/dt =: w2

    D = th2 - th1 # abbreviate

    # d(w1)/dt
    Denom1 = (M1 + M2)*L1 - M2*L1*cos(D)*cos(D)
    dydx[1] = (M2*L1*w1*w1*sin(D)*cos(D) +
               M2*G*sin(th2)*cos(D) +
               M2*L2*w2*w2*sin(D) -
               (M1 + M2)*G*sin(th1)) / Denom1

    # d(w2)/dt
    Denom2 = (L2/L1)*Denom1
    dydx[3] = (-M2*L2*w2*w2*sin(D)*cos(D) +
               (M1 + M2)*G*sin(th1)*cos(D) -
               (M1 + M2)*L1*w1*w1*sin(D) -
               (M1 + M2)*G*sin(th2)) / Denom2

    return dydx


# Note: scipy's integrate.odeint can use larger dt,
#       (without leaking energy), but ain't faster.
step = with_rk4(dxdt, autonom=True)


def energy(x):
    """Compute total energy of system."""
    th1, th1d, th2, th2d = x.T
    # Potential
    V = -(M1+M2)*L1*G*np.cos(th1) - M2*L2*G*np.cos(th2)
    # Kinetic
    T = 0.5*M1* (L1*th1d)**2 + \
        0.5*M2*((L1*th1d)**2 + (L2*th2d)**2 + \
        2*L1*L2*th1d*th2d*np.cos(th1-th2))
    # Sum
    return T + V


from dapper.mods.Lorenz63.core import LPs as L63_LPs
LP_setup = lambda jj: L63_LPs(jj, params=dict())



