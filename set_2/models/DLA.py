import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def solve_laplace(grid, phi, omega, tolerance=1e-4, max_iter=100):
    """
    Calculates the probability field (phi) around the cluster using SOR.
    Boundary Conditions: The top of the grid is set to 1 (the source),
    and the cluster itself is set to 0.
    It repeats this until the field stops changing significantly or hits max_iter.

    """
    phi[0, :] = 1.0     # Source
    phi[grid == 1] = 0  # Sink (The Cluster)

    update_mask = (grid == 0)
    update_mask[0, :] = False
    update_mask[-1, :] = False
    iters_taken = max_iter # Default if it doesn't converge early
    
    for i in range(max_iter):
        old_phi = phi.copy()
        
        # Red-Black Gauss-Seidel / SOR update
        for r, c in [(0,0), (1,1), (0,1), (1,0)]:
            sl = (slice(1+r, -1, 2), slice(1+c, -1, 2))
            m = update_mask[sl]
            res = 0.25 * (phi[r:-2:2, 1+c:-1:2] + phi[2+r::2, 1+c:-1:2] +
                          phi[1+r:-1:2, c:-2:2] + phi[1+r:-1:2, 2+c::2])
            phi[sl][m] = (1 - omega) * phi[sl][m] + omega * res[m]

        if np.max(np.abs(phi - old_phi)) < tolerance:
            iters_taken = i + 1
            break
            
    return phi, iters_taken


def find_growth_candidates(grid: np.array):
    """
    Returns an array where a value of True represents a growth candidate.
    """
    # Create a boolean mask of the cluster
    cluster = grid.astype(bool)
    
    # Initialize a mask for neighbors (same size as grid)
    neighbors = np.zeros_like(cluster)
    
    neighbors[1:]  |= cluster[:-1]  # Shift Down 
    neighbors[:-1] |= cluster[1:]   # Shift Up 
    neighbors[:, 1:]  |= cluster[:, :-1] # Shift Right 
    neighbors[:, :-1] |= cluster[:, 1:]  # Shift Left 
    
    # Candidates are neighbor cells that are NOT already part of the cluster
    candidates_mask = neighbors & ~cluster
    return candidates_mask

def add_candidates(grid, candidates_mask, phi, eta):
    """
    Looks at the values of phi at the candidate locations. 
    Eta controls how much the field affects growth. 
    Picks one candidate based on these probabilities and sets that cell in the grid to 1.
    """
    candidate_coords = np.argwhere(candidates_mask)
    c_values = np.clip(phi[candidates_mask], 0, 1)
    
    weights = np.power(c_values, eta)
    total_weight = np.sum(weights)

    if total_weight < 1e-15:
        probabilities = np.ones(len(weights)) / len(weights)
    else:
        # Standardize probabilities for np.random.choice
        probabilities = weights / total_weight
        probabilities /= probabilities.sum() 
    
    chosen_idx = np.random.choice(len(candidate_coords), p=probabilities)
    target = candidate_coords[chosen_idx]
    
    # Update Grid and Phi simultaneously
    grid[target[0], target[1]] = 1
    phi[target[0], target[1]] = 0 # Immediately make it a sink
    
    return grid, phi


def DLA(size=100, iterations=100, eta=1, omega=1):
    """    
    1. start with a single seed at the bottom of the domain
    2. solve the laplace eq
    3. locate growth candidates around the cluster and assign a growth prob for each candidate
    4. add a growth candidate to the cluster with prob p_g
    5. solve laplace again, repeat
    """
    grid = np.zeros((size, size))
    grid[-1, size // 2] = 1 # Seed
    
    phi = np.zeros((size, size))
    phi[0, :] = 1.0 
    phi[-1, size // 2] = 0 # Seed is a sink

    for i in range(iterations):
        # Warm Start solve
        phi, x = solve_laplace(grid, phi, tolerance=1e-4, max_iter=60, omega=omega)
        
        candidates = find_growth_candidates(grid)
        # Pass phi in and get the updated phi back
        grid, phi = add_candidates(grid, candidates, phi, eta)
        
        if i % 100 == 0:
            print(f"Iteration {i} complete...")
            
    return grid