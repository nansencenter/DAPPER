# Find Lyapunov exponents of L63 system: 0.88, 0, -14.54.
# Based on (book) Lynch - Dynamical Systems with Matlab;
# program 14d: Lyapunov exponents of a Lorenz system,
# which is based on
# A. Wolf, J. B. Swift, H. L. Swinney, and J. A. Vastano,
# "Determining Lyapunov Exponents from a Time Series," Physica D.

# The (eps-scaled) ensemble version,
# which works without an analytic tangent linear model,
# is made by intuition.

from common import *
from scipy.integrate import odeint, ode

T  = 100
dt = 0.02
K  = round(T/dt)
tt = linspace(dt,T,K)

m   = 3 # ndim

eps = 0.002 # ensemble rescaling


########################
# Model
########################

# Constants
sig = 10; rho = 28; beta = 8/3

def dxdt(x):
  """
  Full (nonlinear) model.
  Works for ensemble input (as well).
  """
  d = np.zeros_like(x)
  x = x.T
  d = d.T
  d[0] = sig*(x[1] - x[0])
  d[1] = rho*x[0] - x[1] - x[0]*x[2]
  d[2] = x[0]*x[1] - beta*x[2]
  return d.T

#vec = lambda U : U .ravel()
#mat = lambda uu: uu.reshape((m,m))

def dUdt(U,x):
  """Tangent-Linear-Model"""
  x,y,z = x
  TLM=[[-sig, sig, 0],
      [rho-z, -1, -x],
      [y, x, -beta]]
  return TLM@U


### Doesn't work coz difficult to pass-in the reference trajectory:
#init   : solver = ode(lambda t,x: dUdt(x)).set_integrator('dopri5')
#iterate: U = solver.set_initial_value(U, t).integrate(t+dt)
### Doesn't work coz requires reshaping mat/vec:
#U = odeint(lambda x,t: dUdt(x,Ref), U, [t, t+dt])[1]
def step_U(U,dt,Ref):
  return rk4(lambda t,x: dUdt(x,Ref), U, np.nan, dt)

def step_x(x0, dt):
  return rk4(lambda t,x: dxdt(x), x0, np.nan, dt)


########################
# Main loop
########################
L_exp_TLM = zeros((K,m))
L_exp_Ens = zeros((K,m))

# Init
x = ones(m)
U = eye(m)
E = x + eps*U

for k,t in enumerate(tt):
  # t_now = t - dt
  x = step_x(x,dt)

  U            = step_U(U,dt,x)
  [U, F]       = sla.qr(U)
  L_exp_TLM[k] = log(np.abs(diag(F)))

  E            = step_x(E,dt)
  E            = (E-x).T/eps
  [Q, R]       = sla.qr(E)
  E            = x + eps*Q.T
  L_exp_Ens[k] = log(np.abs(diag(R)))

# Running averages
avrg_TLM = ( np.cumsum(L_exp_TLM,axis=0).T / tt ).T
avrg_Ens = ( np.cumsum(L_exp_Ens,axis=0).T / tt ).T

print("norm of diff TLM-Ens of BLVs(T): ", sla.norm(U-Q,'fro')/sqrt(m))


########################
# Plot
########################
plt.clf()
plt.plot(tt,avrg_TLM,lw=2)
plt.plot(tt,avrg_Ens,'--')
plt.ylim([-15,5])
plt.title('Lyapunov Exponent estimates')
plt.xlabel('Time')
plt.ylabel('Exponent value')
for i in range(m):
  l = avrg_TLM[-1][i]
  plt.text(0.7*T,l+0.3,
      '$\lambda = {:.5g}$'.format(l) ,size=12)



