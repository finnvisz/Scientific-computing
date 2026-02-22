"""
Plotting utilities for wave, diffusion, and Laplace results.
"""
# TODO: move from notebook: plot_concentration_profiles, plot_final_states,
#       wave animation setup; any other figure/axis helpers

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.animation as animation

# -------- Wave Equationn (1.1) --------

def init_wave_plot(show_plot=True):
    """Create 1x3 figure and axes for wave equation (your notebook style)."""
    if not show_plot:
        return None, None
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    return fig, axes


def update_wave_plot(axes, x, psi_i, psi_ii, psi_iii, color):
    """Add one snapshot (three curves) with given color (your notebook style)."""
    if axes is None:
        return
    axes[0].plot(x, psi_i, color=color, alpha=0.8)
    axes[1].plot(x, psi_ii, color=color, alpha=0.8)
    axes[2].plot(x, psi_iii, color=color, alpha=0.8)


def finalize_wave_plot(fig, axes, cmap, norm, savepath=None):
    """Set titles, colorbar, tight_layout, optionally save, then show (your notebook style)."""
    if axes is None:
        return
    axes[0].set_title(r'$\sin(2\pi x)$')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel(r'$\psi_i^{n+1}$')
    axes[1].set_title(r'$\sin(5\pi x)$')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel(r'$\psi_i^{n+1}$')
    axes[2].set_title(r'$\sin(5\pi x)$ for $1/5 < x < 2/5$')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel(r'$\psi_i^{n+1}$')
    plt.tight_layout()
    plt.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), ax=axes, label='Time')
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()

def create_wave_animation(x, stored_i, stored_ii, stored_iii,
                          titles=None, ylim=(-1, 1), interval=150, savepath=None):
    """
    Create animation from stored wave frames.
    If savepath is set (e.g. .gif or .mp4), the animation is saved to file.
    """
    if titles is None:
        titles = [
            r'$\sin(2\pi x)$',
            r'$\sin(5\pi x)$',
            r'$\sin(5\pi x)$ for $1/5 < x < 2/5$'
        ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, title in zip(axes, titles):
        ax.set_ylim(ylim)
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel(r'$\psi_i^{n+1}$')

    line_i, = axes[0].plot(x, stored_i[0], 'b-', lw=2)
    line_ii, = axes[1].plot(x, stored_ii[0], 'b-', lw=2)
    line_iii, = axes[2].plot(x, stored_iii[0], 'b-', lw=2)

    def update(frame):
        line_i.set_ydata(stored_i[frame])
        line_ii.set_ydata(stored_ii[frame])
        line_iii.set_ydata(stored_iii[frame])
        return line_i, line_ii, line_iii

    anim = animation.FuncAnimation(
        fig, update, frames=len(stored_i),
        interval=interval, blit=True
    )

    if savepath is not None:
        # .gif: writer='pillow' or 'imagemagick'; .mp4: writer='ffmpeg'
        try:
            anim.save(savepath, writer='pillow')
        except Exception:
            anim.save(savepath, writer='ffmpeg')
    return anim


# -------- Time-dependent 2D diffusion (1.2) --------

def plot_diffusion_numerical_vs_analytic(y_coords, data_list, savepath=None):
    """
    Plot numerical vs analytic solution at several times (E).
    data_list: list of (c_slice, c_analytic, t) for each target time.
    Migrated from time_dep_diff_eq.py as-is.
    """
    plt.figure(figsize=(10, 6))
    for c_slice, c_analytic, t in data_list:
        line, = plt.plot(y_coords, c_slice, 'o', markersize=4, label=f'Numerical t={t:.3f}')
        plt.plot(y_coords, c_analytic, color=line.get_color(), linestyle='-', alpha=0.6, label=f'Analytic')
    plt.title('Comparison of Numerical vs. Analytic Solutions')
    plt.xlabel('y')
    plt.ylabel('Concentration c(y)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()


def plot_diffusion_2d_snapshot(c, t, savepath=None):
    """
    Single 2D concentration snapshot (F). extent [0,1,0,1], origin lower, viridis.
    Migrated from time_dep_diff_eq.py as-is.
    """
    plt.imshow(c, extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
    plt.title(f"Time t = {t:.3f}")
    plt.colorbar()
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()


def create_diffusion_animation(step_fn, c_init, r, N, frames=200, steps_per_frame=5, interval=50, savepath=None):
    """
    Animation of 2D diffusion evolution (G). step_fn(c_curr, r, N) returns c_next.
    Migrated from time_dep_diff_eq.py as-is (5 steps per frame, 200 frames).
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    c_holder = [c_init.copy()]
    im = ax.imshow(c_holder[0], extent=[0, 1, 0, 1], origin='lower', cmap='viridis', animated=True)
    ax.set_title("Evolution to Equilibrium")
    fig.colorbar(im)

    def animate(frame):
        for _ in range(steps_per_frame):
            c_holder[0] = step_fn(c_holder[0], r, N)
        im.set_array(c_holder[0])
        return [im]

    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=True)
    if savepath is not None:
        try:
            anim.save(savepath, writer='pillow', fps=30)
        except Exception:
            anim.save(savepath, writer='ffmpeg', fps=30)
    return anim

# -------- Time-independent Laplace / iterative (1.3–1.6) --------

def plot_state_array(M, title=None, savepath=None):
    """
    Plot 2D state array (imshow, gray, colorbar). Migrated from time_independent.py as-is.
    """
    plt.imshow(M, cmap='gray')
    plt.colorbar()
    if title is not None:
        plt.title(title)
    if savepath is not None:
        plt.savefig(savepath)
    else:
        plt.show()
    plt.close()


def plot_error(errs, title=None, savepath=None):
    """
    Plot error vs iteration (log scale). Migrated from time_independent.py as-is.
    """
    plt.title(title)
    plt.xlabel("k")
    plt.ylabel("$\delta$")
    plt.yscale("log")
    plt.plot(errs)
    if savepath is not None:
        plt.savefig(savepath)
    else:
        plt.show()
    plt.close()

# -------- Question K: iteration and optimal-omega sweeps --------

def plot_question_k_iteration_sweeps(
    sizes_cntr,
    gauss_iterations_cntr,
    sor_iterations_cntr,
    num_rects_list,
    gauss_iterations_multiple,
    sor_iterations_multiple,
    savepath=None,
):
    """
    Two-panel plot: Convergence vs rectangle size (objects_in_cntr) and vs number of rectangles (objects_in_domain_multiple).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(sizes_cntr, gauss_iterations_cntr, 'o-', label='Gauss-Seidel', linewidth=2, markersize=8)
    axes[0].plot(sizes_cntr, sor_iterations_cntr, 's-', label='SOR (ω=1.9)', linewidth=2, markersize=8)
    axes[0].set_xlabel('Rectangle Size (tolerance)', fontsize=12)
    axes[0].set_ylabel('Iterations to Convergence', fontsize=12)
    axes[0].set_title('Convergence vs Rectangle Size\n(objects_in_cntr)', fontsize=13)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(sizes_cntr)
    axes[1].plot(num_rects_list, gauss_iterations_multiple, 'o-', label='Gauss-Seidel', linewidth=2, markersize=8)
    axes[1].plot(num_rects_list, sor_iterations_multiple, 's-', label='SOR (ω=1.9)', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Rectangles', fontsize=12)
    axes[1].set_ylabel('Iterations to Convergence', fontsize=12)
    axes[1].set_title('Convergence vs Number of Rectangles\n(objects_in_domain_multiple)', fontsize=13)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(num_rects_list)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()
    plt.close()


def plot_question_k_optimal_omega_sweeps(
    sizes_cntr,
    optimal_omegas_cntr,
    num_rects_list,
    optimal_omegas_multiple,
    savepath=None,
):
    """
    Two-panel plot: Optimal ω vs rectangle size and vs number of rectangles (with ω=1.9 reference line).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(sizes_cntr, optimal_omegas_cntr, 'o-', linewidth=2, markersize=8, color='purple')
    axes[0].set_xlabel('Rectangle Size (tolerance)', fontsize=12)
    axes[0].set_ylabel('Optimal ω', fontsize=12)
    axes[0].set_title('Optimal ω vs Rectangle Size\n(objects_in_cntr)', fontsize=13)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(sizes_cntr)
    axes[0].axhline(y=1.9, color='r', linestyle='--', alpha=0.5, label='ω=1.9 (default)')
    axes[0].legend(fontsize=10)
    axes[1].plot(num_rects_list, optimal_omegas_multiple, 'o-', linewidth=2, markersize=8, color='purple')
    axes[1].set_xlabel('Number of Rectangles', fontsize=12)
    axes[1].set_ylabel('Optimal ω', fontsize=12)
    axes[1].set_title('Optimal ω vs Number of Rectangles\n(objects_in_domain_multiple)', fontsize=13)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(num_rects_list)
    axes[1].axhline(y=1.9, color='r', linestyle='--', alpha=0.5, label='ω=1.9 (default)')
    axes[1].legend(fontsize=10)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()
    plt.close()
