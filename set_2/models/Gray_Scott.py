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
    

    # Small amount of random noise
    u_grid += 0.01 * np.random.randn(N, N)
    v_grid += 0.01 * np.random.randn(N, N)

    return u_grid, v_grid

def A_matrix(N, alpha):
    N_total = N * N
    A = lil_matrix((N_total, N_total))

    for i in range(N):
        for j in range(N):
            k_index = i * N + j
            jp1, jm1 = (j + 1) % N, (j - 1) % N
            ip1, im1 = (i + 1) % N, (i - 1) % N
            k_right = i * N + jp1
            k_left = i * N + jm1
            k_down = ip1 * N + j
            k_up = im1 * N + j

            A[k_index, k_index] = 1 + 2 * alpha
            A[k_index, k_right] = -alpha / 2
            A[k_index, k_left] = -alpha / 2
            A[k_index, k_down] = -alpha / 2
            A[k_index, k_up] = -alpha / 2

    A = A.tocsc()
    return A

def b_vector(N, alpha, dt, f, k, u_profile, v_profile, U=True):
    N_total = N * N
    b = np.zeros(N_total)

    for i in range(N):
        for j in range(N):
            k_index = i * N + j
            jp1, jm1 = (j + 1) % N, (j - 1) % N
            ip1, im1 = (i + 1) % N, (i - 1) % N

            if U:
                diffusion_term = (u_profile[i, j] * (1 - 2 * alpha)) + (alpha / 2) * (
                    u_profile[ip1, j] + u_profile[im1, j] + u_profile[i, jp1] + u_profile[i, jm1]
                )
                reaction_term = dt * (
                    -u_profile[i, j] * (v_profile[i, j]) ** 2 + f * (1 - u_profile[i, j])
                )
            else:
                diffusion_term = (v_profile[i, j] * (1 - 2 * alpha)) + (alpha / 2) * (
                    v_profile[ip1, j] + v_profile[im1, j] + v_profile[i, jp1] + v_profile[i, jm1]
                )
                reaction_term = dt * (
                    u_profile[i, j] * (v_profile[i, j]) ** 2 - (f + k) * v_profile[i, j]
                )

            b[k_index] = diffusion_term + reaction_term

    return b




