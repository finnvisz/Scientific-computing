"""
Time-independent 2D diffusion / Laplace (1.3-1.6).
Initial state (state_array), convergence threshold, SOR iteration count (sor_iterations).

"""
import numpy as np
from scientific_computing_set1.solvers.iterative import successive_over_relaxation_step

conv_threshold = 1e-5


def state_array(N, random_seed=None):
    """Initial grid: zeros with top row 1; optional random init. Full BCs: top=1, bottom=0, periodic sides."""
    M = np.zeros((N, N))
    if random_seed is not None:
        np.random.seed(random_seed)
        M += np.random.uniform(0, 1, size=(N, N))
    # boundary conditions (match time_independent.py)
    M[0, :] = 1
    M[-1, :] = 0
    M[:, 0] = M[:, -1]
    return M


def sor_iterations(omega, N=50, max_iter=10000, threshold=conv_threshold):
    """Run SOR with a given omega and return the number of iterations to converge."""
    M = state_array(N)
    for k in range(1, max_iter + 1):
        M_old = np.copy(M)
        M = successive_over_relaxation_step(M, omega)
        if np.max(np.abs(M - M_old)) < threshold:
            return k
    return max_iter

def sor_iterations_with_history(omega, N=50, max_iter=10000, threshold=conv_threshold, random_seed=None):
    """Run SOR with a given omega; return (M, k, errs) for convergence-rate plotting and random-init runs."""
    M = state_array(N, random_seed=random_seed)
    errs = []
    for k in range(1, max_iter + 1):
        M_old = np.copy(M)
        M = successive_over_relaxation_step(M, omega)
        err = np.max(np.abs(M - M_old))
        errs.append(err)
        if err < threshold:
            return M, k, errs
    return M, max_iter, errs

# Objects in the computational domain:
def objects_in_cntr(N, y_tolerance, x_tolerance):
    """
    Creates a square object in the center of the computational domain.
    
    Parameters:
    -----------
    N : int
        Grid size
    y_tolerance : int
        Tolerance in the y-direction
    x_tolerance : int
        Tolerance in the x-direction
    Returns:
    --------
    M : array
        Array with the object in the center of the computational domain
    """
    M = np.zeros((N, N))
    # A sync at the center of the computational domain, representing the object
    M[(N//2) - y_tolerance :(N//2) + y_tolerance, (N//2) - x_tolerance :N//2 + x_tolerance] = 1
    
    return M

# Multiple rectangles in the computational domain:
def objects_in_domain_multiple(N, rectangles):
    """
    rectangles: list of (center_y, center_x, y_tolerance, x_tolerance) for each rectangle.
    Same convention as objects_in_cntr: rectangle spans
    [center - tolerance, center + tolerance) in each dimension.
    """
    M = np.zeros((N, N))
    for rect in rectangles:
        cy, cx, y_tolerance, x_tolerance = rect
        M[cy - y_tolerance : cy + y_tolerance, cx - x_tolerance : cx + x_tolerance] = 1
    return M

def rectangles_for_count(N, num_rects):
    """Return list of rectangles in a matrix configuration for this many rectangles (Question K sweep 2)."""
    grid_cols = int(np.ceil(np.sqrt(num_rects)))
    grid_rows = int(np.ceil(num_rects / grid_cols))
    x_spacing = N // (grid_cols + 1)
    y_spacing = N // (grid_rows + 1)
    rectangles = []
    rect_count = 0
    for row in range(grid_rows):
        for col in range(grid_cols):
            if rect_count >= num_rects:
                break
            x_pos = x_spacing * (col + 1)
            y_pos = y_spacing * (row + 1)
            rectangles.append((y_pos, x_pos, 3, 2))
            rect_count += 1
        if rect_count >= num_rects:
            break
    return rectangles

# -------- Concentration field formulation (Question K iterative solvers) --------

def initialize_concentration_field(N):
    """
    Initialize concentration field with boundary conditions.
    c[:, 0] = 0 (bottom), c[:, N-1] = 1 (top).
    """
    c = np.zeros((N, N))
    c[:, 0] = 0      # bottom boundary (y=0)
    c[:, N-1] = 1    # top boundary (y=1)
    return c


def jacobi_iteration(N, epsilon=1e-5, M=None):
    """
    Run Jacobi iteration until convergence. Optional object mask M (M[i,j]==1 -> c=0).
    Returns (iterations, final_state, concentration_profiles).
    """
    initial_c = initialize_concentration_field(N)
    new_c = np.zeros((N, N))
    if M is None:
        M = np.zeros((N, N))
    iterations = 0
    concentration_profiles = []
    while True:
        new_c[:, 1:-1] = (1/4) * (
            np.roll(initial_c[:, 1:-1], -1, axis=0) + np.roll(initial_c[:, 1:-1], 1, axis=0) +
            initial_c[:, 2:] + initial_c[:, :-2]
        )
        new_c[:, 0] = 0
        new_c[:, N-1] = 1
        new_c[M == 1] = 0
        error = np.max(np.abs(new_c - initial_c))
        if iterations % 30 == 0:
            concentration_profiles.append(new_c[1, :].copy())
        if error < epsilon:
            break
        iterations += 1
        initial_c = np.copy(new_c)
    return iterations, new_c, concentration_profiles


def gauss_seidel_iteration(N, epsilon=1e-5, M=None):
    """
    Run Gauss-Seidel until convergence. Optional mask M (M[i,j]==1 -> c=0).
    Returns (iterations, final_state, concentration_profiles).
    """
    new_c = initialize_concentration_field(N)
    if M is None:
        M = np.zeros((N, N))
    iterations = 0
    concentration_profiles = []
    while True:
        max_dif = 0
        for i in range(N):
            for j in range(1, N-1):
                if M[i, j] == 1:
                    old_value = new_c[i, j]
                    new_c[i, j] = 0
                else:
                    old_value = new_c[i, j]
                    new_c[i, j] = (1/4) * (
                        new_c[(i+1) % N, j] + new_c[(i-1) % N, j] +
                        new_c[i, j+1] + new_c[i, j-1]
                    )
                if np.abs(new_c[i, j] - old_value) > max_dif:
                    max_dif = np.abs(new_c[i, j] - old_value)
        if iterations % 30 == 0:
            concentration_profiles.append(new_c[1, :].copy())
        if max_dif < epsilon:
            break
        iterations += 1
    return iterations, new_c, concentration_profiles


def sor_iteration(N, omega=1.9, epsilon=1e-5, M=None):
    """
    Run SOR until convergence. Optional mask M (M[i,j]==1 -> c=0).
    Returns (iterations, final_state, concentration_profiles).
    """
    new_c = initialize_concentration_field(N)
    if M is None:
        M = np.zeros((N, N))
    iterations = 0
    concentration_profiles = []
    while True:
        max_dif = 0
        for i in range(N):
            for j in range(1, N-1):
                if M[i, j] == 1:
                    old_value = new_c[i, j]
                    new_c[i, j] = 0
                else:
                    old_value = new_c[i, j]
                    new_c[i, j] = (omega/4) * (
                        new_c[(i+1) % N, j] + new_c[(i-1) % N, j] +
                        new_c[i, j+1] + new_c[i, j-1]
                    ) + (1-omega) * new_c[i, j]
                if np.abs(new_c[i, j] - old_value) > max_dif:
                    max_dif = np.abs(new_c[i, j] - old_value)
        if iterations % 30 == 0:
            concentration_profiles.append(new_c[1, :].copy())
        if max_dif < epsilon:
            break
        iterations += 1
    return iterations, new_c, concentration_profiles