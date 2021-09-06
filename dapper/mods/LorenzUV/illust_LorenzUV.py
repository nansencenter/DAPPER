"""Prettier, static illustration of Lorenz two-speed/scale/layer model."""
# Sorry for the mess.

if __name__ == "__main__":
    import matplotlib as mpl
    import numpy as np
    from matplotlib import pyplot as plt

    import dapper.mods as modelling
    from dapper.mods.LorenzUV.lorenz96 import LUV

    # Setup
    sd0 = modelling.set_seed(4)
    # from dapper.mods.LorenzUV.wilks05 import LUV
    nU, J = LUV.nU, LUV.J

    dt = 0.005
    t0 = np.nan
    K  = int(10/dt)

    step_1 = modelling.with_rk4(LUV.dxdt, autonom=True)
    step_K = modelling.with_recursion(step_1, prog=1)

    x0 = 0.01*np.random.randn(LUV.M)
    x0 = step_K(x0, int(2/dt), t0, dt)[-1]  # BurnIn
    xx = step_K(x0, K, t0, dt)

    # Grab parts of state vector
    ii = np.arange(nU+1)
    jj = np.arange(nU*J+1)
    circU = np.mod(ii, nU)
    circV = np.mod(jj, nU*J) + nU
    iU = np.hstack([0, 0.5+np.arange(nU), nU])

    def Ui(xx):
        interp = (xx[0]+xx[-1])/2
        return np.hstack([interp, xx, interp])

    # Overlay linear
    fig, ax = plt.subplots()
    L = 20  # Num of lines to plot
    start = int(3e5*dt)
    step  = 3
    for i, Ny in enumerate(range(L)):
        k = start + Ny*step
        c = mpl.cm.viridis(1-Ny/L)
        a = 0.8-0.2*Ny/L
        plt.plot(iU, Ui(xx[k][:nU]), color=c, lw=2, alpha=a)[0]
        if i % 2 == 0:
            plt.plot(jj/J, xx[k][circV], color=c, lw=0.7, alpha=a)[0]
    # Make ticks, ticklabels, grid
    ax.set_xticks([])
    ym, yM = -4, 10
    ax.set_ylim(ym, yM)
    ax.set_xlim(0, nU)
    dY = 4  # SET TO: 1 for wilks05, 4 for lorenz96
    # U-vars: major
    tU = iU[1:-1]
    lU = np.array([str(i+1) for i in range(nU)])
    tU = np.concatenate([[tU[0]], tU[dY-1::dY]])
    lU = np.concatenate([[lU[0]], lU[dY-1::dY]])
    for t, l in zip(tU, lU):
        ax.text(t, ym-.6, l,
                fontsize=mpl.rcParams['xtick.labelsize'], horizontalalignment='center')
        ax.vlines(t, ym, -3.78, 'k', lw=mpl.rcParams['xtick.major.width'])
    # V-vars: minor
    tV = np.arange(nU+1)
    lV = ['1'] + [str((i+1)*J) for i in circU]
    for i, (t, l) in enumerate(zip(tV, lV)):
        if i % dY == 0:
            ax.text(t, -5.0, l, fontsize=9, horizontalalignment='center')
            ax.vlines(t, ym, yM, lw=0.3)
        ax.vlines(t, ym, -3.9, 'k', lw=mpl.rcParams['xtick.minor.width'])
    ax.grid(color='k', alpha=0.6, lw=0.4, axis='y', which='major')

    # Convert to circular coordinates
    # Should have used instead: projection='polar'
    def tU(zz):
        xx  = (40 + 3*zz)*np.cos(2*np.pi*ii/nU)
        yy  = (40 + 3*zz)*np.sin(2*np.pi*ii/nU)
        return xx, yy

    def tV(zz):
        xx  = (80 + 15*zz)*np.cos(2*np.pi*jj/nU/J)
        yy  = (80 + 15*zz)*np.sin(2*np.pi*jj/nU/J)
        return xx, yy

    # # Animate circ
    # plt.subplots()
    # lhU   = plt.plot(*tU(xx[-1][circU]),'b',lw=3)[0]
    # lhV   = plt.plot(*tV(xx[-1][circV]),'g',lw=1)[0]
    # from dapper.tools.progressbar import progbar
    # for k in progbar(range(K),'Plotting'):
    #     dataU = tU(xx[k][circU])
    #     dataV = tV(xx[k][circV])
    #     lhU.set_xdata(dataU[0])
    #     lhU.set_ydata(dataU[1])
    #     lhV.set_xdata(dataV[0])
    #     lhV.set_ydata(dataV[1])
    #     plt.pause(0.001)

    ## Logo -- Overlay circ
    fig, ax = plt.subplots()
    plt.plot(*tU(4.52*np.ones_like(circU)), color='k', lw=1)[0]
    plt.plot(*tV(0.15*np.ones_like(circV)), color='k', lw=1)[0]
    ax = fig.axes[0]
    ax.set_axis_off()
    ax.set_facecolor('white')
    ax.set_aspect('equal')
    L = 40  # Num of lines to plot
    for Ny in range(L):
        k = 143 + Ny*3
        c = mpl.cm.viridis(1-Ny/L)
        a = 0.8-0.2*Ny/L
        plt.plot(*tU(xx[k][circU]), color=c, lw=2, alpha=a)[0]
        plt.plot(*tV(xx[k][circV]), color=c, lw=1, alpha=a)[0]

    # Add DAPPER text label
    if False:
        ax.text(95, 0, "DAPPER", ha="left", va="center", fontdict=dict(
            fontsize="80", name="Signpainter"))

        # Adjust xlim to include everything
        ax.set_xlim((-100, 400))

        # bbox_inches="tight" doesnt manage with this font,
        # so plot something almost invisible
        ax.plot([348, 348.001], [28, 28])

    if False:
        fig.savefig("docs/imgs/logo_wtxt.png",
                    bbox_inches="tight",
                    pad_inches=0,
                    dpi=200)

    plt.show()
