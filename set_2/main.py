"""Imports"""
import numpy as np
import matplotlib.pyplot as plt

from set_2.models.DLA import (
    DLA,
    solve_laplace,
    find_growth_candidates,
    add_candidates
)

def main():
    """Run all assignment questions."""
    # Plot the growth for different values of eta
    size=100
    etas = [0.2, 0.5, 1, 1.5, 2]
    iterations=1000
    omega = 1.7

    fig, axes = plt.subplots(1, len(etas), figsize=(5*len(etas), 5))

    for ax, eta in zip(axes, etas):
        result = DLA(size=100, iterations=iterations, eta=eta, omega=omega)
        
        ax.imshow(result, cmap='Blues')
        ax.set_title(f"η={eta}", fontsize=25)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(f"DLA_eta_comparison_omega={omega}.png")

    # Find optimal omega
    omegas = np.arange(1.0, 2.0, 0.05)
    num_trials = 1  # How many full DLA growths to run per omega
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
    plt.savefig("result_stats_omega.png")


if __name__ == "__main__":
    main()