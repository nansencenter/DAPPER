"""Compare integration time-stepping schemes for KS equation.

Conclusions: 'ETD-RK4' is superior to 'SI-RK3',

- Superiority deteriorates when adding noise to ICs.
- Superiority fairly independent of N.
- The naive methods (rk4, rk1) are quite terrible.
"""

if __name__ == "__main__":
    import time

    import numpy as np
    from matplotlib import pyplot as plt

    from dapper.mods.KS import Model

    model = Model()

    print("THERE WILL BE WARNINGS GENERATED.")
    print("This is normal, because some schemes cause NaNs for some dt's.")

    # Experiment range
    minexpo = -8  # use -11 to observe saturation due to num. prec.
    hh = 2.0**np.arange(minexpo, 0)  # list of dt

    # Params
    T = 30
    N = 4

    # IC -- NB: integration accuracy depends on noise level
    E0 = model.x0_Kassam + 1e-3*np.random.randn(N, model.Nx)

    # Allocate stats
    mref        = 'step_ETD_RK4'
    methods     = [mref]
    methods    += [key for key in model if key.startswith("step_") and key != mref]
    NamedFloats = np.dtype([(m, float) for m in methods])
    errors      = np.zeros(len(hh), dtype=NamedFloats)
    durations   = np.zeros(len(hh), dtype=NamedFloats)

    for i, dt in enumerate(hh):

        # Model for this dt.
        model = Model(dt=dt)

        for m in methods:
            E = E0.copy()
            t0 = time.time()
            # Integration
            for _ in 1+np.arange(int(round(T/dt))):
                E = model[m](E, np.nan, dt)
            durations[m][i] = time.time() - t0

            # Compare with reference
            if m == mref and dt == hh[0]:
                assert dt == min(hh), "Must define ref from smallest dt."
                ref = E
            errors[m][i] = np.max(abs(ref - E))

    # Plot stats
    plt.figure()
    for m in methods:
        plt.loglog(durations[m], errors[m], '-o', label=m)
        plt.text(durations[m][0], errors[m][0], 'dt min')
        # plt.text  (durations[m][-1], errors[m][-1], 'dt max')
    plt.legend()
    plt.xlabel('Total computation time (s)')
    plt.ylabel('Max err (vs ref) at T=%d' % T)
    plt.grid(True, 'minor')
    plt.title('Ens size: %d' % N)
    plt.show()

    # Plot ultimate states -- Not interesting, except for debugging
    # plt.figure()
    # plt.plot(grid,E.T)
