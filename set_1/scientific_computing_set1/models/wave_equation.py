"""
Wave equation (1.1): vibrating string.
Initial conditions, time stepping, simulation run, animation.
"""
# TODO: move from notebook / vibrating_string: create_initial_conditions, wave_time_step,
#       run_wave_simulation (plot + store frames), create_wave_animation

import numpy as np
from scientific_computing_set1.solvers.time_stepping import wave_first_step, wave_time_step
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scientific_computing_set1.utils.plotting import (
    init_wave_plot,
    update_wave_plot,
    finalize_wave_plot,
)


# Parameters
L = 1.0 # length of string
c = 1.0
dt = 0.001 # timestep size
N = 100 # string devided in N intervals
dx = L / N
x = np.linspace(0, L, N+1)

def simulate(initial_psi, timesteps=1000):
    psi_past = np.array(initial_psi)
    psi_now = np.zeros_like(psi_past)

    wave_first_step(psi_past, psi_now, c, dt, N)

    # Time stepping
    results = [psi_past.copy()]
    for n in range(timesteps):
        psi_next = np.zeros_like(psi_now)
        wave_time_step(psi_past, psi_now, psi_next, c, dt, N)

        psi_past[:] = psi_now
        psi_now[:] = psi_next

        if n % 100 == 0: # Save every 100 steps for plotting
            results.append(psi_now.copy())
    return results

def create_initial_conditions(x):
    """Create the three initial conditions for the wave equation (your notebook)."""
    psi_i = np.sin(2 * np.pi * x)
    psi_ii = np.sin(5 * np.pi * x)
    psi_iii = np.where((x > 1/5) & (x < 2/5), np.sin(5 * np.pi * x), 0)
    return psi_i, psi_ii, psi_iii

def run_wave_simulation(x, dt, c, N, plot_every=450, frame_skip=100, show_plot=True, timesteps=1000, savepath=None):
    """
    Run wave equation for all three ICs with your notebook-style plotting.
    Uses partner's wave_first_step / wave_time_step. Returns stored frames.
    """
    psi_i, psi_ii, psi_iii = create_initial_conditions(x)
    psi_past_i = np.copy(psi_i)
    psi_past_ii = np.copy(psi_ii)
    psi_past_iii = np.copy(psi_iii)
    psi_now_i = np.zeros_like(psi_i)
    psi_now_ii = np.zeros_like(psi_ii)
    psi_now_iii = np.zeros_like(psi_iii)

    wave_first_step(psi_past_i, psi_now_i, c, dt, N)
    wave_first_step(psi_past_ii, psi_now_ii, c, dt, N)
    wave_first_step(psi_past_iii, psi_now_iii, c, dt, N)

    fig, axes = init_wave_plot(show_plot)
    if show_plot:
        norm = mcolors.Normalize(vmin=0, vmax=1)
        cmap = plt.cm.viridis
    else:
        norm = cmap = None

    # Time-stepping loop
    time = np.arange(1, N, dt)
    n_steps = len(time)

    stored_i = []
    stored_ii = []
    stored_iii = []

    for i, t in enumerate(time):
        psi_next_i = np.zeros_like(psi_now_i)
        psi_next_ii = np.zeros_like(psi_now_ii)
        psi_next_iii = np.zeros_like(psi_now_iii)
        wave_time_step(psi_past_i, psi_now_i, psi_next_i, c, dt, N)
        wave_time_step(psi_past_ii, psi_now_ii, psi_next_ii, c, dt, N)
        wave_time_step(psi_past_iii, psi_now_iii, psi_next_iii, c, dt, N)

        psi_past_i[:] = psi_now_i
        psi_now_i[:] = psi_next_i
        psi_past_ii[:] = psi_now_ii
        psi_now_ii[:] = psi_next_ii
        psi_past_iii[:] = psi_now_iii
        psi_now_iii[:] = psi_next_iii

        if show_plot and i % plot_every == 0:
            color = cmap(norm(i / n_steps))
            update_wave_plot(axes, x, psi_now_i, psi_now_ii, psi_now_iii, color)
        if i % frame_skip == 0:
            stored_i.append(psi_now_i.copy())
            stored_ii.append(psi_now_ii.copy())
            stored_iii.append(psi_now_iii.copy())

    if show_plot:
        finalize_wave_plot(fig, axes, cmap, norm, savepath=savepath)
    return stored_i, stored_ii, stored_iii