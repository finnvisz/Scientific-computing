# Gray Scott Reaction-Diffusion Model

import numpy as np
from scipy.sparse import lil_matrix
from numba import jit

def grid_initialization(N, y_tolerance, x_tolerance, u_initial, v_initial):
    """
    Initializes values in the computational domain, with the U concentration everywhere, with the exception of 
    the V concentration in the center of the computational domain, which is represented by a square.
    
    Parameters:
    -----------
    N : int
        grid size
    y_tolerance : int
        Square size tolerance in the y-direction
    x_tolerance : int
        Square size tolerance in the x-direction
    u_initial : float
        Initial concentration of U in the computational domain
    v_initial : float
        Initial concentration of V in the center square in the computational domain
    Returns:
    --------
    grid : 2D array of shape NxN
        Array with the initial V concentration in the center of the computational domain
    """

    # Initialize U chemical and V chemical grids
    u_grid = np.zeros((N,N))
    u_grid[:,:] = u_initial

    v_grid = np.zeros((N,N))
    v_grid[(N//2) - y_tolerance :(N//2) + y_tolerance, (N//2) - x_tolerance :N//2 + x_tolerance] = v_initial
    

    # Small amount of random noise
    u_grid += 0.01 * np.random.randn(N, N)
    v_grid += 0.01 * np.random.randn(N, N)

    return u_grid, v_grid

def A_matrix(grid_size, sigma):
    """
    Construct the sparse matrix A for the Crank-Nicolson scheme with periodic boundaries.
    
    The matrix implements a 5-point stencil for the 2D Laplacian with periodic boundary
    conditions. Each row corresponds to one grid point and contains 5 non-zero entries:
    - Diagonal entry: (1 + 2·σ)
    - Four neighbors: -σ/2 each
    
    Parameters
    ----------
    grid_size : int
        Number of grid points in each direction (N·N grid)
    sigma : float
        Diffusion parameter σ = D·Δt/Δx²
        
    Returns
    -------
    scipy.sparse.csc_matrix
        Sparse matrix A of size (N²·N²) in CSC format
    """
    N = grid_size
    N_total = N * N
    
    # Use LIL format for efficient construction
    A = lil_matrix((N_total, N_total))
    
    # Loop over all grid points
    for i in range(N):
        for j in range(N):
            # Current point's flattened index
            k_center = i * N + j
            
            # Compute periodic neighbor indices in 2D
            j_right = (j + 1) % N
            j_left = (j - 1) % N
            i_down = (i + 1) % N
            i_up = (i - 1) % N
            
            # Convert neighbor indices to flattened 1D indices
            k_right = i * N + j_right
            k_left = i * N + j_left
            k_down = i_down * N + j
            k_up = i_up * N + j
            
            # Fill matrix row k with 5-point stencil coefficients
            A[k_center, k_center] = 1 + 2 * sigma      # Center (diagonal)
            A[k_center, k_right] = -sigma / 2          # Right neighbor
            A[k_center, k_left] = -sigma / 2           # Left neighbor
            A[k_center, k_down] = -sigma / 2           # Down neighbor
            A[k_center, k_up] = -sigma / 2             # Up neighbor
    
    # Convert to CSC format for efficient sparse linear algebra
    return A.tocsc()

def b_vector(N, alpha, dt, f, k, u_profile, v_profile, U=True):
    """
    Construct the right-hand side vector b for the Crank-Nicolson scheme with periodic boundaries.
    
    The vector implements a 5-point stencil for the 2D Laplacian with periodic boundary
    conditions. Each entry corresponds to one grid point and contains 5 non-zero entries:
    - Diagonal entry: (1 + 2·σ)
    - Four neighbors: -σ/2 each
    
    Parameters
    ----------
    N : int
        Number of grid points in each direction (N·N grid)
    alpha : float
        Diffusion parameter σ = D·Δt/Δx²
    dt : float
        Time step
    f : float
    k : float
        Rate constant for the reaction
    u_profile : 2D array of shape NxN
        U concentration profile
    v_profile : 2D array of shape NxN
        V concentration profile
    U : bool
        True if U is the target chemical, False if V is the target chemical
    Returns
    -------
    b : 1D array of shape N²·N²
        Right-hand side vector b of size (N²·N²) in CSC format
    """
    u_up = np.roll(u_profile, 1, axis=0)
    u_down = np.roll(u_profile, -1, axis=0)
    u_left = np.roll(u_profile, 1, axis=1)
    u_right = np.roll(u_profile, -1, axis=1)
    v_up = np.roll(v_profile, 1, axis=0)
    v_down = np.roll(v_profile, -1, axis=0)
    v_left = np.roll(v_profile, 1, axis=1)
    v_right = np.roll(v_profile, -1, axis=1)

    if U:
        diffusion = u_profile * (1 - 2 * alpha) + (alpha / 2) * (u_up + u_down + u_left + u_right)
        reaction = dt * (-u_profile * v_profile**2 + f * (1 - u_profile))
    else:
        diffusion = v_profile * (1 - 2 * alpha) + (alpha / 2) * (v_up + v_down + v_left + v_right)
        reaction = dt * (u_profile * v_profile**2 - (f + k) * v_profile)

    return (diffusion + reaction).ravel()




