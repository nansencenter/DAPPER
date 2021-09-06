"""Estimtate the Lyapunov spectrum and other props. of the dynamics of the models."""

# An old version using explicit TLMs can be found in EmblAUS/Lyap_L{63,96}.

if __name__ == "__main__":
    import numpy as np
    import scipy.linalg as sla
    from matplotlib import pyplot as plt
    from mpl_tools import place
    from numpy.random import randn

    import dapper.tools.series as series
    import dapper.tools.viz as viz
    from dapper.mods import set_seed, with_recursion, with_rk4
    from dapper.tools.progressbar import progbar

    set_seed(3000)

    ########################
    # Model selection
    ########################
    mod = "L05"

    # Lyapunov exponents: [ 1.05  0.   -0.01 -1.05]
    if mod == "DP":
        from dapper.mods.DoublePendulum import step, x0
        T   = 5e2
        dt  = 0.005
        eps = 0.0002
        Nx  = len(x0)
        N   = Nx

    # Lyapunov exponents: [0.02  0 -0.28 -1.03]
    if mod == "LV":
        from dapper.mods.LotkaVolterra import step, x0
        T   = 1e3
        dt  = 0.2
        eps = 0.0002
        Nx  = len(x0)
        N   = Nx

    # Lyapunov exponents: [ 0.51 -0.72]
    if mod == "Ikeda":
        from dapper.mods.Ikeda import step, x0
        T   = 5e3
        dt  = 1
        eps = 1e-5
        Nx  = len(x0)
        N   = Nx

    # Lyapunov exponents: [0.906, 0, -14.572]
    if mod == "L63":
        from dapper.mods.Lorenz63 import step, x0
        T   = 1e2
        dt  = 0.04
        Nx  = len(x0)
        eps = 0.001
        N   = Nx

    # Lyapunov exponents: [ 0.22, 0, -0.52]
    if mod == "L84":
        from dapper.mods.Lorenz84 import step, x0
        T   = 1e3
        dt  = 0.05
        Nx  = len(x0)
        eps = 0.001
        N   = Nx

    # Reproduces findings of Carrassi-2008 "Model error and sequential DA...",
    # when setting M=36, F=8, dt=0.0083="1hour":  LyapExps ∈ ( −0.97, 0.33 ) /hour.
    # In unitless time (as used here), this means LyapExps ∈ ( −4.87, 1.66 ) .
    if mod == "L96":
        from dapper.mods.Lorenz96 import step
        Nx  = 40             # State size (flexible). Usually 36 or 40
        T   = 1e3            # Length of experiment (unitless time).
        dt  = 0.1            # Step length
        # dt = 0.0083        # Any dt<0.1 yield "almost correct" Lyapunov expos.
        x0  = randn(Nx)      # Init condition.
        eps = 0.0002         # Ens rescaling factor.
        N   = Nx             # Num of perturbations used.

    # Lyapunov exponents with F=10:
    # [9.47   9.3    8.72 ..., -33.02 -33.61 -34.79] => n0:64
    if mod == "LUV":
        from dapper.mods.LorenzUV.lorenz96 import LUV
        ii    = np.arange(LUV.nU)
        step  = with_rk4(LUV.dxdt, autonom=True)
        Nx    = LUV.M
        T     = 1e2
        dt    = 0.005
        LUV.F = 10
        x0    = 0.01*randn(LUV.M)
        eps   = 0.001
        N     = 66  # Don't need all Nx for a good approximation of upper spectrum.

    # Lyapunov exponents: [8.36, 7.58, 7.20, 6.91, ..., -4.18, -4.22, -4.19] => n0≈164
    if mod == "L05":
        from dapper.mods.LorenzIII import Model
        model = Model()
        step  = model.step
        x0    = model.x0
        dt    = 0.05/12
        Nx    = len(x0)
        N     = 400
        T     = 1e2
        eps   = 0.0002

    # Lyapunov exponents: [  0.08   0.07   0.06 ... -37.9  -39.09 -41.55]
    if mod == "KS":
        from dapper.mods.KS import Model
        KS   = Model()
        step = KS.step
        x0   = KS.x0
        dt   = KS.dt
        N    = KS.Nx
        Nx   = len(x0)
        T    = 1e3
        eps  = 0.0002

    # n0 ≈ 140
    if mod == "QG":
        from dapper.mods.QG import model_config, sample_filename, shape

        # NB: There may arise an ipython/multiprocessing bug/issue.
        # Ref https://stackoverflow.com/a/45720872 . If so, set mp=False,
        # or run outside of ipython. However, I did not encounter it lately.
        model = model_config("sakov2008", {}, mp=True)
        step  = model.step
        Nx    = np.prod(shape)
        ii    = np.random.choice(np.arange(Nx), 100, False)
        T     = 1000.0
        dt    = model.prms['dtout']
        x0    = np.load(sample_filename)['sample'][-1]
        eps   = 0.01  # ensemble rescaling
        N     = 300

    ########################
    # Reference trajectory
    ########################
    # NB: Arbitrary, coz models are autonom. But dont use nan coz QG doesn't like it.
    t0 = 0.0
    K  = int(round(T/dt))       # Num of time steps.
    tt = np.linspace(dt, T, K)  # Time seq.
    x  = with_recursion(step, prog="BurnIn")(x0, int(10/dt), t0, dt)[-1]
    xx = with_recursion(step, prog="Reference")(x, K, t0, dt)

    ########################
    # ACF
    ########################
    # NB: Won't work with QG (too big, and BCs==0).
    fig, ax = place.freshfig("ACF")
    if "ii" not in locals():
        ii = np.arange(min(100, Nx))
    if "nlags" not in locals():
        nlags = min(100, K-1)
    ax.plot(tt[:nlags], np.nanmean(
        series.auto_cov(xx[:nlags, ii], nlags=nlags-1, corr=1),
        axis=1))
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Auto-corr')
    viz.plot_pause(0.1)

    ########################
    # "Linearized" forecasting
    ########################
    LL = np.zeros((K, N))        # Local (in time) Lyapunov exponents
    E  = x + eps*np.eye(Nx)[:N]  # Init E

    for k, t in enumerate(progbar(tt, "Ens (≈TLM)")):
        # if t%10.0==0: print(t)

        x      = xx[k+1]  # = step(x,t,dt)    # f.cast reference
        E      = step(E, t, dt)               # f.cast ens (i.e. perturbed ref)

        E      = (E-x).T/eps                  # Compute f.cast perturbations
        [Q, R] = sla.qr(E, mode='economic')   # Orthogonalize
        E      = x + eps*Q.T                  # Init perturbations
        LL[k]  = np.log(np.abs(np.diag(R)))   # Store local Lyapunov exponents

    ########################
    # Running averages
    ########################
    running_LS = (1/tt[:, None] * np.cumsum(LL, axis=0))
    LS = running_LS[-1]
    print('Lyapunov spectrum estimate after t=T:')
    with np.printoptions(precision=2):
        print(LS)
    n0 = sum(LS >= 0)
    print('n0  : ', n0)
    print('var : ', np.var(xx))
    print('mean: ', np.mean(xx))

    ##

    fig, ax = place.freshfig("Lyapunov exponents")
    ax.plot(tt, running_LS, lw=1.2, alpha=0.7)
    ax.axhline(0, c="k")
    ax.set_title('Lyapunov Exponent estimates')
    ax.set_xlabel('Time')
    ax.set_ylabel('Exponent value')
    # Annotate values
    # for L in LS:
    #     ax.text(0.7*T,L+0.01, '$\lambda = {:.5g}$'.format(L) ,size=12)
    plt.show()
