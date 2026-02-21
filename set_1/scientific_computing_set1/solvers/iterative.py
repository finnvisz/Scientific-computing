"""
Iterative solvers for the 2D Laplace equation (steady-state diffusion).
Jacobi, Gauss-Seidel, and SOR (Successive Over-Relaxation).
"""
# TODO: move from notebook: jacobi_iteration, gauss_seidel_iteration, sor_iteration
# Optional object mask M (M[i,j]==1 -> concentration 0). Return (iterations, final_state, profiles).

import numpy as np


def jacobi_step(M):
    """One Jacobi step (migrated from time_independent.py as-is)."""
    new_M = np.copy(M)
    N = M.shape[0]
    for i in range(1, N-1):
        new_M[i, :] = 0.25 * (M[i-1, :] + M[i+1, :] + np.roll(M[i, :], 1) + np.roll(M[i, :], -1))
    return new_M


def gauss_seidel_step(M):
    """One Gauss-Seidel step (migrated from time_independent.py as-is)."""
    N = M.shape[0]
    for i in range(1, N-1):
        for j in range(N):
            M[i, j] = 0.25 * (M[i-1, j] + M[i+1, j] + M[i, j-1] + M[i, (j+1)%N])
    return M


def successive_over_relaxation_step(M, omega):
    """One SOR step (migrated from time_independent.py as-is)."""
    N = M.shape[0]
    for i in range(1, N-1):
        for j in range(N):
            M[i, j] = (omega/4) * (M[i-1, j] + M[i+1, j] + M[i, j-1] + M[i, (j+1)%N]) + (1 - omega) * M[i, j]
    return M
