# Gray Scott Reaction-Diffusion Model

import numpy as np
from scipy.sparse import lil_matrix


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
    

    # Small noise helps patterns develop
    u_grid += 0.01 * np.random.randn(N, N)
    v_grid += 0.01 * np.random.randn(N, N)

    return u_grid, v_grid

def A_matrix(N, alpha):

    N_total = N * N
    A = lil_matrix((N_total, N_total))

    for i in range(N):
        for j in range(N):
            k_index = i * N + j

            if i > 0 and i < N-1 and j > 0 and j < N-1:
                A[k_index, k_index] = 1 + 2 * alpha
                A[k_index, k_index + 1] = -alpha/2
                A[k_index, k_index - 1] = -alpha/2
                A[k_index, k_index + N] = -alpha/2
                A[k_index, k_index - N] = -alpha/2
            
            else:
                A[k_index, k_index] = 1  # Identity: u_new[boundary] = u_old[boundary]

    A = A.tocsc() # Converts to Compressed Sparse Column (CSC) format for efficient linear algebra operations

    return A

def b_vector(N, alpha, dt, f, k, u_profile, v_profile, U=True):

    N_total = N * N
    b = np.zeros(N_total)

    for i in range(N):
        for j in range(N):
            k_index = i * N + j

            if i > 0 and i < N-1 and j > 0 and j < N-1:
                if U:
                    diffusion_term = (u_profile[i,j] *(1 - 2 * alpha)) + (alpha/2) * (u_profile[i+1,j] + u_profile[i-1,j] + u_profile[i,j+1] + u_profile[i,j-1])
                    reaction_term = dt * (- u_profile[i,j] * (v_profile[i,j])**2 + f * (1 - u_profile[i,j]))

                    b[k_index] = diffusion_term + reaction_term

                else:
                    diffusion_term = (v_profile[i,j] *(1 - 2 * alpha)) + (alpha/2) * (v_profile[i+1,j] + v_profile[i-1,j] + v_profile[i,j+1] + v_profile[i,j-1])
                    reaction_term = dt * (u_profile[i,j] * (v_profile[i,j])**2 - (f + k) * v_profile[i,j])

                b[k_index] = diffusion_term + reaction_term
            
            else:
                if U:
                    b[k_index] = u_profile[i,j]
                else:
                    b[k_index] = v_profile[i,j]

    return b




