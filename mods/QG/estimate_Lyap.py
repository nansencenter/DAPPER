# Estimtate the Lyapunov spectrum of the QG model,
# using a limited (rank-N) ensemble.
# Inspired by EmblAUS/Lor95_Lyap.py

from common import *
import mods.QG.core as mod

sd0 = seed(5)

eps = 0.01 # ensemble rescaling

T  = 600.0
dt = 5.0
K  = round(T/dt)
tt = linspace(dt,T,K)

m  = mod.m # ndim

def step_x(x0):
  return mod.step(x0, 0.0, dt)

########################
# Main loop
########################

x = mod.flatten(mod.psi0)

# Init U
N = 300
U = eye(m)[:N]
E = x + eps*U

LL_exp = zeros((K,N))

for k,t in enumerate(tt):
  if t%10.0==0: print(t)
  # t_now = t - dt
  x = step_x(x)

  E         = step_x(E)
  E         = (E-x).T/eps
  [Q, R]    = sla.qr(E,mode='economic')
  E         = x + eps*Q.T
  LL_exp[k] = log(abs(diag(R)))


# Running averages
running_LS = ( tp(1/tt) * np.cumsum(LL_exp,axis=0) )
LS         = running_LS[-1]
print('Lyap spectrum estimate at t=T:')
with printoptions(precision=4): print(LS)
n0 = sum(LS >= 0)
print('n0: ', n0)


#########################
## Plot
#########################
plt.clf()
plt.plot(tt,running_LS)
plt.title('Lyapunov Exponent estimates')
plt.xlabel('Time')
plt.ylabel('Exponent value')





