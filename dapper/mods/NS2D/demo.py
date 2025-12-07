import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from dapper.mods.NS2D import Model

model = Model(N=64, dt=0.01, T=200, nu=1 / 1600)
T = model.T
tt = np.linspace(0, T, int(T / model.dt), endpoint=True, dtype=np.float64)
EE = np.zeros((len(tt), model.Nx * model.Nx))
EE[0] = model.x0.flatten()  # IC comes from model; change IC to change demo output

for k in tqdm.tqdm(range(1, len(tt))):
    EE[k] = model.step(EE[k - 1], np.nan, model.dt)


def animate_snapshots(psis, snapshot_steps):
    # Select n evenly spaced indices (n = snapshot_steps)
    indices = np.linspace(0, len(psis), snapshot_steps, False, dtype=int)
    psi_snapshots = psis[indices]
    # For title, get the actual time step for each snapshot
    step_numbers = indices

    # Animate
    fig, ax = plt.subplots()
    im = ax.imshow(
        psi_snapshots[0],
        origin="lower",
        cmap="viridis",
        extent=(0, model.DL * np.pi, 0, model.DL * np.pi),
    )
    ax.set_title(f"Streamfunction ψ, step {step_numbers[0]}")
    plt.colorbar(im, ax=ax)

    def update(frame):
        im.set_data(psi_snapshots[frame])
        ax.set_title(f"Streamfunction ψ, step {step_numbers[frame]}")
        return (im,)

    _ = animation.FuncAnimation(
        fig, update, frames=len(psi_snapshots), interval=100, blit=False
    )
    plt.show()


animate_snapshots(EE.reshape(len(tt), model.Nx, model.Nx), 10)
