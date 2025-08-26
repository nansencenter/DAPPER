import numpy as np
from dapper.mods.NS2D import Model
import matplotlib.pyplot as plt
import matplotlib.animation as animation


model = Model(dt=0.01, T=200, nu=1/1600)
T = model.T
tt = np.linspace(0, T, int(T / model.dt), endpoint = True, dtype=np.float64)
EE = np.zeros((len(tt), model.Nx, model.Nx))
EE[0] = model.x0
for k in range(1, len(tt)):
    EE[k] = model.step(EE[k - 1], np.nan, model.dt)
def animate_snapshots(psis, snapshot_steps):
        if len(psis) == 0:
            print("No psi snapshots were saved. Animation will not run.")
            return
        # Select evenly spaced indices
        indices = np.linspace(0, len(psis), snapshot_steps, False, dtype=int)
        psi_snapshots = psis[indices]
        # For title, get the actual time step for each snapshot
        step_numbers = indices
        # Check shapes
        shapes = [np.shape(s) for s in psi_snapshots]
        if not all(s == shapes[0] and len(s) == 2 for s in shapes):
            raise ValueError(f"Not all psi_snapshots have the same 2D shape: {shapes}")
        # Animate
        fig, ax = plt.subplots()
        im = ax.imshow(psi_snapshots[0], origin='lower', cmap='viridis', extent=(0, model.DL * np.pi, 0, model.DL * np.pi))
        ax.set_title(f'Streamfunction ψ, step {step_numbers[0]}')
        plt.colorbar(im, ax=ax)
        def update(frame):
            im.set_data(psi_snapshots[frame])
            ax.set_title(f'Streamfunction ψ, step {step_numbers[frame]}')
            return (im,)
        ani = animation.FuncAnimation(fig, update, frames=len(psi_snapshots), interval=100, blit=False)
        plt.show()
animate_snapshots(EE, 10)