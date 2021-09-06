"""Illusrate parameterizations.

Plot scattergram of "unresolved tendency" and the parameterization that emulate it.

We plot the diff:
  `model_step/dt - true_step/dt    (1)`
Whereas Wilks plots
  `model_dxdt    - true_step/dt    (2)`
Another option is:
  `model_dxdt    - true_dxdt       (3)`

Thus, for us (eqn 1), the model integration scheme matters.
Also, Wilks uses
- `dt = 0.001` for truth
- `dt = 0.005` for model.
"""

if __name__ == "__main__":
    import matplotlib as mpl
    import numpy as np
    from matplotlib import pyplot as plt

    import dapper.mods as modelling
    from dapper.mods import with_recursion, with_rk4

    # mpl.rcParams['toolbar'] = 'None'
    ###########################
    # Setup
    ###########################
    PRMS = 'Lorenz'
    if PRMS == 'Wilks':
        from dapper.mods.LorenzUV.wilks05 import LUV
    else:
        from dapper.mods.LorenzUV.lorenz96 import LUV
    nU = LUV.nU

    K  = 400
    dt = 0.005
    t0 = np.nan

    modelling.set_seed(30)  # 3 5 7 13 15 30
    x0 = np.random.randn(LUV.M)

    true_step  = with_rk4(LUV.dxdt, autonom=True)
    model_step = with_rk4(LUV.dxdt_trunc, autonom=True)
    true_K     = with_recursion(true_step, prog=1)

    ###########################
    # Compute truth trajectory
    ###########################
    x0 = true_K(x0, int(2/dt), t0, dt)[-1]  # BurnIn
    xx = true_K(x0, K, t0, dt)

    ###########################
    # Compute unresovled scales
    ###########################
    gg = np.zeros((K, nU))  # "Unresolved tendency"
    for k, x in enumerate(xx[:-1]):
        X = x[:nU]
        Z = model_step(X, t0, dt)
        D = Z - xx[k+1, :nU]
        gg[k] = 1/dt*D

    # Automated regression for deterministic parameterizations
    pc = {}
    for order in [0, 1, 2, 3, 4]:
        pc[order] = np.polyfit(xx[:-1, :nU].ravel(), gg.ravel(), deg=order)

    ###########################
    # Scatter plot
    ###########################
    def blend_rgb(rgb, a, bg_rgb=(1, 1, 1)):
        """Fake RGB transparency by blending it to some background.

        Useful for creating gradients.

        Also useful for creating 'transparency' for exporting to eps.
        But there's no actualy transparency, so superposition of lines
        will not work. For that: export to pdf, or make do without.

        - `rgb`: N-by-3 rgb, or a color code.
        - `a`: alpha value
        - `bg_rgb`: background in rgb. Default: white

        Based on stackoverflow.com/a/33375738/38281
        """
        if isinstance(rgb, str):
            rgb = mpl.colors.colorConverter.to_rgb(rgb)
        return [a*c1 + (1-a)*c2 for (c1, c2) in zip(rgb, bg_rgb)]

    xx = xx[:-1, :nU]
    dk = int(8/dt/50)  # step size
    xx = xx[::dk].ravel()
    gg = gg[::dk].ravel()

    fig, ax = plt.subplots()
    ax.scatter(xx, gg, facecolors='none', edgecolors=blend_rgb('k', 0.5), s=40)
    # ax.plot(xx,gg,'o',color=[0.7]*3)
    ax.set_ylabel(r'Unresolved tendency ($q_{k,i}/\Delta t$)')
    ax.set_xlabel('Resolved variable ($X_{k,i}$)')

    ###########################
    # Lorenz'96 Parameterization annotations
    ###########################
    if PRMS != 'Wilks':
        ax.set_xlim(-8, 12)
        ax.set_ylim(-3, 6)
        uu = np.linspace(*ax.get_xlim(), 201)

        # Plot pc
        ax.plot(uu, np.poly1d(pc[0])(uu), 'g', lw=4.0)
        ax.plot(uu, np.poly1d(pc[1])(uu), 'r', lw=4.0)
        ax.plot(uu, np.poly1d(pc[4])(uu), 'b', lw=4.0)

    ###########################
    # Wilks2005 Parameterization annotations
    ###########################
    else:
        ax.set_xlim(-10, 17)
        ax.set_ylim(-10, 20)
        uu = np.linspace(*ax.get_xlim(), 201)

        def p0(x): return 3.82+0.00*x
        # lin.reg(gg,xx)
        def p1(x): return 0.74+0.82*x
        # Arnold'2013:
        def p3(x): return .341+1.30*x - .0136*x**2 - .00235*x**3
        # Wilks'2005:
        def p4(x): return .262+1.45*x - .0121*x**2 - .00713*x**3 + .000296*x**4
        uu = np.linspace(-10, 17, 201)
        plt.plot(uu, p0(uu), 'g', lw=4.0)
        plt.plot(uu, p1(uu), 'r', lw=4.0)
        plt.plot(uu, p4(uu), 'b', lw=4.0)
        # plt.plot(uu,p3(uu),'y',lw=3.0)

        def an(T, xy, xyT, HA='left'):
            ah = ax.annotate(T,
                             xy=xy, xycoords='data',
                             xytext=xyT, textcoords='data',
                             fontsize=16,
                             horizontalalignment=HA,
                             arrowprops=dict(
                                 arrowstyle="->",
                                 connectionstyle="arc3", lw=2))
            return ah

        s4 = '$0.262$\n$+1.45X$\n$-0.0121X^2$\n$-0.00713X^3$\n$+0.000296X^4$'
        an('$3.82$', (10, 3.82), (10, -2), 'center')
        an('$0.74+0.82X$', (-7.4, -5.4), (1, -6))
        an(s4, (7, 8), (0, 10), 'right')
