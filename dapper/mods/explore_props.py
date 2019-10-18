# Estimtate the Lyapunov spectrum using an ensemble of perturbations.
# An obsolete version using explicit TLMs can be found in EmblAUS/Lyap_L{63,96}.

from dapper import *
sd0 = seed(5)
plt.ion()

########################
# Model selection
########################


# from dapper.mods.DoublePendulum.core import step, x0
# ------------------
# Lyapunov exponents: [ 1.05  0.   -0.01 -1.05]
# T   = 5e2
# dt  = 0.005
# eps = 0.0002
# Nx  = len(x0)
# N   = Nx

# from dapper.mods.LotkaVolterra.core import step, x0
# # ------------------
# # Lyapunov exponents: [0.02  0 -0.28 -1.03]
# T   = 1e3
# dt  = 0.2
# eps = 0.0002
# Nx  = len(x0)
# N   = Nx


# from dapper.mods.Lorenz63.core import step, x0
# # ------------------
# # Lyapunov exponents: [0.906, 0, -14.572]
# T   = 1e2
# dt  = 0.04
# Nx  = len(x0)
# eps = 0.001
# N   = Nx

# from dapper.mods.Lorenz84.core import step, x0
# # ------------------
# # Lyapunov exponents: [ 0.22, 0, -0.52]
# T   = 1e3
# dt  = 0.05
# Nx  = len(x0)
# eps = 0.001
# N   = Nx

# from dapper.mods.Lorenz96.core import step
# # ------------------
# # Reproduces findings of Carrassi-2008 "Model error and sequential DA...",
# # when setting M=36, F=8, dt=0.0083="1hour":  LyapExps ∈ ( −0.97, 0.33 ) /hour.
# # In unitless time (as used here), this means LyapExps ∈ ( −4.87, 1.66 ) .
# Nx  = 40        # State size (flexible). Usually 36 or 40
# T   = 1e3       # Length of experiment (unitless time).
# dt  = 0.1       # Step length
# # dt = 0.0083     # Any dt<0.1 yield "almost correct" Lyapunov expos.
# x0  = randn(Nx) # Init condition.
# eps = 0.0002    # Ens rescaling factor.
# N   = Nx        # Num of perturbations used.

# from dapper.mods.LorenzUV.lorenz96 import LUV
# # ------------------
# # Lyapunov exponents with F=10: [9.47   9.3    8.72 ..., -33.02 -33.61 -34.79] => n0:64
# ii    = arange(LUV.nU)
# step  = with_rk4(LUV.dxdt, autonom=True)
# Nx    = LUV.M
# T     = 1e2
# dt    = 0.005
# LUV.F = 10
# x0    = 0.01*randn(LUV.M)
# eps   = 0.001
# N     = 66 # Don't need all Nx for a good approximation of upper spectrum.


# from dapper.mods.KS.core import Model
# # ------------------
# # Lyapunov exponents: [  0.08   0.07   0.06 ... -37.9  -39.09 -41.55]
# KS   = Model()
# step = KS.step
# x0   = KS.x0
# dt   = KS.dt
# N    = KS.Nx
# Nx   = len(x0)
# T    = 1e3
# eps  = 0.0002


# from dapper.mods.QG.core import model_config, shape, sample_filename
# # ------------------
# # n0 ≈ 140
# # NB: "Sometimes" multiprocessing does not work here.
# # This may be an ipython bug (stackoverflow.com/a/45720872).
# # Solutions: 1) run script from outside of ipython,
# #         or 2) Turn it off using mp=False.
# model = model_config("sakov2008",{},mp=False)
# step  = model.step
# Nx    = prod(shape)
# ii    = np.random.choice(arange(Nx),100,False)
# T     = 1000.0
# dt    = model.prms['dtout']
# x0    = np.load(sample_filename)['sample'][-1]
# eps   = 0.01 # ensemble rescaling
# N     = 300



########################
# Reference trajectory
########################
t0 = 0.0 # NB: Arbitrary, coz models are autonom. But dont use nan coz QG doesn't like it.
K  = int(round(T/dt))                      # Num of time steps.
tt = linspace(dt,T,K)                      # Time seq.
x  = with_recursion(step, prog="BurnIn")   (x0, int(10/dt), t0, dt)[-1]
xx = with_recursion(step, prog="Reference")(x , K,          t0, dt)


########################
# ACF
########################
# NB: Won't work with QG (too big, and BCs==0).
fig, ax = freshfig(4)
if "ii"    not in locals(): ii    = arange(min(100,Nx))
if "nlags" not in locals(): nlags = min(100,K)
ax.plot(tt[:nlags], np.nanmean( auto_cov(xx[:nlags,ii],L=nlags,corr=1), axis=1) )
ax.set_xlabel('Time (t)')
ax.set_ylabel('Auto-corr')
plot_pause(0.1)


########################
# "Linearized" forecasting
########################
LL = zeros((K,N))        # Local (in time) Lyapunov exponents
E  = x + eps*eye(Nx)[:N] # Init E

for k,t in enumerate(progbar(tt,"Ens (≈TLM)")):
    # if t%10.0==0: print(t)

    x     = xx[k+1] # = step(x,t,dt)  # f.cast reference
    E     = step(E,t,dt)              # f.cast ens (i.e. perturbed ref)

    E     = (E-x).T/eps               # Compute f.cast perturbations
    [Q,R] = sla.qr(E,mode='economic') # Orthogonalize
    E     = x + eps*Q.T               # Init perturbations  
    LL[k] = log(abs(diag(R)))         # Store local Lyapunov exponents


########################
# Running averages
########################
running_LS = ( 1/tt[:,None] * np.cumsum(LL,axis=0) )
LS = running_LS[-1]
print('Lyapunov spectrum estimate after t=T:')
with printoptions(precision=2): print(LS)
n0 = sum(LS >= 0)
print('n0  : ', n0)
print('var : ', np.var(xx))
print('mean: ', np.mean(xx))

##

fig, ax = freshfig(1)
ax.plot(tt,running_LS,lw=1.2,alpha=0.7)
ax.set_title('Lyapunov Exponent estimates')
ax.set_xlabel('Time')
ax.set_ylabel('Exponent value')
# Annotate values
# for L in LS: ax.text(0.7*T,L+0.01, '$\lambda = {:.5g}$'.format(L) ,size=12)

##





