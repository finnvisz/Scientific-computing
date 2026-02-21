"""
Time-dependent 2D diffusion (1.2).
Parameters, initial grid, analytic solution (migrated from time_dep_diff_eq.py as-is).
"""
import numpy as np
from scipy.special import erfc
from scientific_computing_set1.solvers.time_stepping import diffusion_2d_step

# Parameters (from time_dep_diff_eq.py as-is)
N = 50
D = 1.0
dx = 1.0 / N
# Stability: dt <= dx^2 / 4D
dt = 0.25 * (dx**2) / D
r = (dt * D) / (dx**2)

# y_coords for 1D slice / analytic (same as script)
y_coords = np.linspace(0, 1, N + 1)

# Default times to plot (E and F)
times_to_plot = [0, 0.001, 0.01, 0.1, 1.0]

def initial_concentration_grid(N):
    """Initial 2D grid: zeros with top boundary c[N, :] = 1.0 (migrated as-is)."""
    c = np.zeros((N + 1, N + 1))
    c[N, :] = 1.0  # Top boundary condition
    return c


def analytic_solution(y, t, D, terms=100):
    """Analytic solution (migrated from time_dep_diff_eq.py as-is)."""
    if t == 0:
        return np.zeros_like(y)
    res = np.zeros_like(y)
    for i in range(terms):
        res += erfc((1 - y + 2 * i) / (2 * np.sqrt(D * t))) - erfc(
            (1 + y + 2 * i) / (2 * np.sqrt(D * t))
        )
    return res

def run_comparison_simulation(times_to_plot=times_to_plot, max_t=1.01):
    """
    Run diffusion to max_t; at each target time collect (c_slice, c_analytic, t).
    Returns (y_coords, data_list) for plot_diffusion_numerical_vs_analytic (E).
    """
    c = initial_concentration_grid(N)
    curr_t = 0.0
    data_list = []
    while curr_t <= max_t:
        for t_target in times_to_plot:
            if np.isclose(curr_t, t_target, atol=dt / 2):
                c_slice = c[:, N // 2].copy()
                c_analytic = analytic_solution(y_coords, curr_t, D)
                data_list.append((c_slice, c_analytic, curr_t))
        c = diffusion_2d_step(c, r, N)
        curr_t += dt
    return y_coords, data_list


def run_2d_snapshots(times_to_plot=times_to_plot, max_t=1.0):
    """
    Run diffusion to max_t; at each target time save (c, t).
    Returns list of (c, t) for plot_diffusion_2d_snapshot (F).
    """
    c = initial_concentration_grid(N)
    curr_t = 0.0
    snapshots = []
    while curr_t <= max_t:
        for t_target in times_to_plot:
            if np.isclose(curr_t, t_target, atol=dt / 2):
                snapshots.append((c.copy(), curr_t))
        c = diffusion_2d_step(c, r, N)
        curr_t += dt
    return snapshots