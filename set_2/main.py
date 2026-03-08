"""Imports"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from mpl_toolkits.axes_grid1 import make_axes_locatable

from set_2.models.DLA import (
    DLA,
    solve_laplace,
    find_growth_candidates,
    add_candidates
)

from set_2.models.Gray_Scott import (
    grid_initialization,
    A_matrix,
    b_vector
)

def main():
    """Run all assignment questions."""
    out = "set_2/outputs"

    ######################
    ###### 2.1 DLA #######
    ######################

    # Plot the growth for different values of eta
    size = 100
    etas = [0, 0.2, 0.5, 1, 1.5, 2, 3, 5, 7, 10]
    iterations = 1000
    omega = 1.7

    rows, cols = 2, 5
    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
    axes_flat = axes.flatten()

    for i, eta in enumerate(etas):
        ax = axes_flat[i]
        result = DLA(size=size, iterations=iterations, eta=eta, omega=omega)
        
        ax.imshow(result, cmap='Blues')
        ax.set_title(f"η={eta}", fontsize=25)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(f'{out}/DLA_eta_comparison_omega={omega}.png')

    # Find optimal omega
    omegas = np.arange(1.0, 2.0, 0.05)
    num_trials = 1  # How many full DLA growths to run per omega (for now set to 1 to decrease runtime)
    results_stats = {}

    for w in omegas:
        trial_means = []
        print(f"Testing Omega {w:.2f}...")
        
        for trial in range(num_trials):
            grid = np.zeros((size, size))
            grid[-1, size // 2] = 1 
            phi = np.zeros((size, size))
            
            iteration_history = []
            for i in range(iterations):
                phi, count = solve_laplace(grid, phi, omega=w, max_iter=200)
                iteration_history.append(count)
                
                candidates = find_growth_candidates(grid)
                grid, phi = add_candidates(grid, candidates, phi, eta=1.0)
            
            trial_means.append(np.mean(iteration_history))
        
        # Average across all trials for this specific omega
        results_stats[w] = np.mean(trial_means)

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(list(results_stats.keys()), list(results_stats.values()), marker='o', linestyle='-', color='b')
    plt.xlabel("Relaxation Parameter (ω)")
    plt.ylabel("Mean Iterations per Step")
    plt.title("SOR Convergence Speed vs. Omega")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig(f'{out}/result_stats_omega.png')

    ##########################
    ## 2.3 Gray-Scott model ##
    ##########################

    dx = 1.0
    dt = 1.0
    D_u = 0.16
    D_v = 0.08
    N = 128
    max_steps = 5000
    plot_steps = [500, 1000, 2500, 5000]

    # (f, k, label) for each run; label used in saved filename
    param_sets = [
        (0.037, 0.060, "Default Parameters"),
        (0.0393, 0.059, "Random Parameters"),
        (0.0545, 0.062, "Coral Growth Parameters"),
    ]

    alpha_u = D_u * dt / dx**2
    alpha_v = D_v * dt / dx**2
    A_u = A_matrix(grid_size=N, sigma=alpha_u)
    A_v = A_matrix(grid_size=N, sigma=alpha_v)

    for f, k, label in param_sets:
        print(f"Gray-Scott: f={f}, k={k} ({label})")
        u_grid, v_grid = grid_initialization(N=N, y_tolerance=5, x_tolerance=5, u_initial=0.5, v_initial=0.25)
        snapshots_u = []
        snapshots_v = []
        iterations = 0

        while iterations < max_steps:
            b_u = b_vector(N=N, alpha=alpha_u, dt=dt, f=f, k=k, u_profile=u_grid, v_profile=v_grid, U=True)
            b_v = b_vector(N=N, alpha=alpha_v, dt=dt, f=f, k=k, u_profile=u_grid, v_profile=v_grid, U=False)
            x = spsolve(A_u, b_u)
            v = spsolve(A_v, b_v)
            u_grid = x.reshape((N, N))
            v_grid = v.reshape((N, N))
            iterations += 1

            if iterations in plot_steps:
                snapshots_u.append(u_grid.copy())
                snapshots_v.append(v_grid.copy())

        n_snap = len(snapshots_u)
        fig2, axes2 = plt.subplots(2, n_snap, figsize=(4 * n_snap, 8))
        for j in range(n_snap):
            im0 = axes2[0, j].imshow(snapshots_u[j], cmap='viridis', vmin=0, vmax=1, aspect='equal')
            axes2[0, j].set_title(f't = {plot_steps[j]}')
            axes2[0, j].set_xlabel('x')
            axes2[0, j].set_ylabel('y')
            div0 = make_axes_locatable(axes2[0, j])
            cax0 = div0.append_axes("right", size="5%", pad=0.05)
            fig2.colorbar(im0, cax=cax0, label='U')

            im1 = axes2[1, j].imshow(snapshots_v[j], cmap='viridis', vmin=0, vmax=1, aspect='equal')
            axes2[1, j].set_title(f't = {plot_steps[j]}')
            axes2[1, j].set_xlabel('x')
            axes2[1, j].set_ylabel('y')
            div1 = make_axes_locatable(axes2[1, j])
            cax1 = div1.append_axes("right", size="5%", pad=0.05)
            fig2.colorbar(im1, cax=cax1, label='V')

        axes2[0, 0].set_ylabel('U-Chemical Concentration Profile\nAt Different Time Steps')
        axes2[1, 0].set_ylabel('V-Chemical Concentration Profile\nAt Different Time Steps')
        plt.suptitle(f'Gray-Scott: f={f}, k={k} ({label})')
        plt.tight_layout()
        plt.savefig(f'{out}/gray_scott_concentration_profiles_{label}.png')


if __name__ == "__main__":
    main()