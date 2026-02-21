"""
Time-stepping utilities for PDEs (e.g. wave equation, time-dependent diffusion).
"""
# TODO: move from notebook / vibrating_string / time_dep_diff_eq: time step logic, frame storage

import numpy as np


def wave_first_step(psi_past, psi_now, c, dt, N):
    """First time step (migrated from vibrating_string.py as-is)."""
    for i in range(1, N):
        psi_now[i] = psi_past[i] + 0.5 * (c * dt *N)**2 * (psi_past[i+1] - 2*psi_past[i] + psi_past[i-1])
    return psi_now


def wave_time_step(psi_past, psi_now, psi_next, c, dt, N):
    """One time step (migrated from vibrating_string.py as-is)."""
    for i in range(1, N):
        psi_next[i] = 2*psi_now[i] - psi_past[i] + (c * dt *N)**2 * (psi_now[i+1] - 2*psi_now[i] + psi_now[i-1])

    # Boundary conditions (Fixed ends)
    psi_next[0] = 0
    psi_next[N] = 0
    return psi_next

def diffusion_2d_step(c_curr, r, N):
    """
    One time step for 2D diffusion with periodic x, fixed top (c[N,:]=1).
    Migrated from time_dep_diff_eq.py as-is.
    """
    c_next = c_curr.copy()

    # Interior updates
    c_next[1:N, 1:N] = c_curr[1:N, 1:N] + r * (
        c_curr[1:N, 2:N+1] + c_curr[1:N, 0:N-1] +
        c_curr[2:N+1, 1:N] + c_curr[0:N-1, 1:N] -
        4*c_curr[1:N, 1:N]
    )

    # Periodic Boundary Updates
    j_idx = np.arange(1, N)
    c_next[j_idx, 0] = c_curr[j_idx, 0] + r * (
        c_curr[j_idx, 1] + c_curr[j_idx, N-1] +
        c_curr[j_idx+1, 0] + c_curr[j_idx-1, 0] -
        4*c_curr[j_idx, 0]
    )
    c_next[:, N] = c_next[:, 0]

    return c_next