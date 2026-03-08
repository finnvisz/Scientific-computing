"""Imports"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

from models.DLA import (
    DLA,
    solve_laplace,
    find_growth_candidates,
    add_candidates
)

from models.Gray_Scott import (
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

    # Clean up layout and save
    plt.tight_layout()
    plt.savefig(f"{out}/DLA_eta_comparison_omega={omega}.png")

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
    plt.savefig(f"{out}/result_stats_omega.png")

    ##########################
    ## 2.3 Gray-Scott model ##
    ##########################

    # Parameters
    dx = 1.0
    dt = 1.0
    D_u = 0.16
    D_v = 0.08
    f = 0.037
    k = 0.060

    u_grid, v_grid = grid_initialization(N=128, y_tolerance=5, x_tolerance=5, u_initial=0.5, v_initial=0.25)

    alpha_u = D_u * dt / dx**2
    alpha_v = D_v * dt / dx**2

    A_u = A_matrix(N=128, alpha= alpha_u)
    A_v = A_matrix(N=128, alpha= alpha_v)

    iterations = 0
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    while iterations < 5000:

        b_u = b_vector(N=128, alpha=alpha_u, dt=dt, f=f, k=k, u_profile=u_grid, v_profile=v_grid, U=True)
        b_v = b_vector(N=128, alpha=alpha_v, dt=dt, f=f, k=k, u_profile=u_grid, v_profile=v_grid, U=False)

        
        x = spsolve(A_u, b_u)
        v = spsolve(A_v, b_v)

        u_grid = x.reshape((int(np.sqrt(len(x))), int(np.sqrt(len(x)))))
        v_grid = v.reshape((int(np.sqrt(len(v))), int(np.sqrt(len(v)))))

        iterations += 1

        if iterations % 1000 == 0:
            print(f"Iteration {iterations}: u=[{u_grid.min():.3f}, {u_grid.max():.3f}], "
                  f"v=[{v_grid.min():.3f}, {v_grid.max():.3f}]")
            
            # Plot
            axes[0].clear()
            axes[0].imshow(u_grid, cmap='viridis', vmin=0, vmax=1)
            axes[0].set_title(f'U at iteration {iterations}')
            
            axes[1].clear()
            axes[1].imshow(v_grid, cmap='viridis', vmin=0, vmax=1)
            axes[1].set_title(f'V at iteration {iterations}')
            
            plt.pause(0.01)
    
    plt.savefig(f"{out}/U_and_V.png")




if __name__ == "__main__":
    main()