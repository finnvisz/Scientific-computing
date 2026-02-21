"""
Main script to run all questions (Scientific Computing Set 1).
"""
# Solver and model imports will be added as modules are populated
# from scientific_computing_set1.solvers import ...
# from scientific_computing_set1.models import ...
# from scientific_computing_set1.utils import ...


"""
Main script to run all questions (Scientific Computing Set 1).
"""
from scientific_computing_set1.models.wave_equation import (x,dt,c,N,run_wave_simulation,)
from scientific_computing_set1.utils.plotting import create_wave_animation

from scientific_computing_set1.models.diffusion_2d import (
    N as N_diff,
    r,
    initial_concentration_grid,
    run_comparison_simulation,
    run_2d_snapshots,
)
from scientific_computing_set1.solvers.time_stepping import diffusion_2d_step
from scientific_computing_set1.utils.plotting import (
    create_wave_animation,
    plot_diffusion_numerical_vs_analytic,
    plot_diffusion_2d_snapshot,
    create_diffusion_animation,
)

import numpy as np
import matplotlib.pyplot as plt
from scientific_computing_set1.models.laplace import (
    conv_threshold,
    state_array,
    sor_iterations,
)
from scientific_computing_set1.solvers.iterative import (
    jacobi_step,
    gauss_seidel_step,
    successive_over_relaxation_step,
)
from scientific_computing_set1.utils.convergence import golden_section_search
from scientific_computing_set1.utils.plotting import (
    create_wave_animation,
    plot_diffusion_numerical_vs_analytic,
    plot_diffusion_2d_snapshot,
    create_diffusion_animation,
    plot_state_array,
    plot_error,
)

from scientific_computing_set1.models.laplace import (
    conv_threshold,
    state_array,
    sor_iterations,
    objects_in_cntr,
    objects_in_domain_multiple,
    rectangles_for_count,
    gauss_seidel_iteration,
    sor_iteration,
)

from scientific_computing_set1.utils.plotting import (
    create_wave_animation,
    plot_diffusion_numerical_vs_analytic,
    plot_diffusion_2d_snapshot,
    create_diffusion_animation,
    plot_state_array,
    plot_error,
    plot_question_k_iteration_sweeps,
    plot_question_k_optimal_omega_sweeps,
)


def main():
    """Run all assignment questions."""
    # 1.1 Wave equation (vibrating string)
    stored_i, stored_ii, stored_iii = run_wave_simulation(
        x, dt, c, N,
        plot_every=450,
        frame_skip=100,
        show_plot=True,
        savepath="scientific_computing_set1/outputs/figures/wave_equation.png"
    )

    anim = create_wave_animation(
        x, stored_i, stored_ii, stored_iii,
        savepath="scientific_computing_set1/outputs/figures/wave_animation.gif",
    )

    # 1.2 Time-dependent diffusion
    out = "scientific_computing_set1/outputs/figures"
    # (E) Numerical vs analytic
    y_coords, data_list = run_comparison_simulation()
    plot_diffusion_numerical_vs_analytic(
        y_coords, data_list,
        savepath=f"{out}/numerical_vs_analytic.png",
    )
    # (F) 2D snapshots at selected times
    snapshots = run_2d_snapshots()
    for c_snap, t in snapshots:
        plot_diffusion_2d_snapshot(
            c_snap, t,
            savepath=f"{out}/diffusion_2D_t={t:.3f}.png",
        )
    # (G) Animation
    c_init = initial_concentration_grid(N_diff)
    create_diffusion_animation(
        diffusion_2d_step, c_init, r, N_diff,
        frames=200, steps_per_frame=5, interval=50,
        savepath=f"{out}/evolution_to_equilibrium.gif",
    )

    # 1.3–1.6 Time-independent diffusion (Laplace, iterative solvers)
    N_laplace = 50
    max_iter = 10000

    # Jacobi
    M = state_array(N_laplace)
    k = 0
    errs = []
    while k < max_iter:
        k += 1
        M2 = jacobi_step(M)
        max_diff = np.max(np.abs(M2 - M))
        errs.append(max_diff)
        if max_diff < conv_threshold:
            break
        M = M2
    print(f"Jacobi converged after {k} steps")
    plot_state_array(M, title="Jacobi", savepath=f"{out}/jacobi.png")
    plot_error(errs, title="Jacobi Error by iteration", savepath=f"{out}/jacobi_error.png")

    # Gauss-Seidel
    M = state_array(N_laplace)
    k = 0
    errs = []
    while k < max_iter:
        k += 1
        M_old = np.copy(M)
        M = gauss_seidel_step(M)
        max_diff = np.max(np.abs(M - M_old))
        errs.append(max_diff)
        if max_diff < conv_threshold:
            break
    print(f"Gauss-Seidel converged after {k} steps")
    plot_state_array(M, title="Gauss-Seidel", savepath=f"{out}/gauss_seidel.png")
    plot_error(errs, title="Gauss-Seidel Error by iteration", savepath=f"{out}/gauss_seidel_error.png")

    # SOR for several omegas
    errs_sor = {}
    omegas = [1.7, 1.8, 1.9, 1.99]
    for omega in omegas:
        M = state_array(N_laplace)
        k = 0
        errs_sor[omega] = []
        while k < max_iter:
            k += 1
            M_old = np.copy(M)
            M = successive_over_relaxation_step(M, omega)
            max_diff = np.max(np.abs(M - M_old))
            errs_sor[omega].append(max_diff)
            if max_diff < conv_threshold:
                break
        print(f"Successive Over Relaxation ($\\omega={omega}$) converged after {k} steps")
        plot_state_array(M, title=f"Successive Over Relaxation, $\\omega={omega}$", savepath=f"{out}/successive_over_relaxation_{omega}.png")

    # Golden section search for optimal omega
    gs_optimal_omega = golden_section_search(sor_iterations, 1.7, 2.0)
    print(f"Golden Search Optimal omega: {gs_optimal_omega:.4f}")
    print(f"Iterations at GS optimal omega: {sor_iterations(gs_optimal_omega)}")
    gs_omega = gs_optimal_omega

    # SOR with optimal omega
    M = state_array(N_laplace)
    gs_errs = []
    for k in range(1, max_iter):
        M_old = np.copy(M)
        M = successive_over_relaxation_step(M, gs_omega)
        max_diff = np.max(np.abs(M - M_old))
        gs_errs.append(max_diff)
        if max_diff < conv_threshold:
            break
    print(f"Successive Over Relaxation ($\\omega={gs_omega:.4f}$) converged after {k} steps")
    plot_state_array(M, title=f"Successive Over Relaxation, Optimal $\\omega={gs_omega:.4f}$", savepath=f"{out}/successive_over_relaxation_optimal.png")

    # SOR error comparison (all omegas + optimal)
    plt.title("Successive Over Relaxation Error")
    plt.xlabel("k")
    plt.ylabel("Error")
    for omega in omegas:
        plt.plot(errs_sor[omega], label=f"$\\omega={omega}$")
    plt.plot(gs_errs, label=f"$\\omega={gs_omega:.4f}$ (optimal)", linestyle='--')
    plt.yscale("log")
    plt.legend()
    plt.savefig(f"{out}/successive_over_relaxation_error.png")
    plt.close()

    # Sweep over N: optimal omega and iterations vs N
    ns = [5, 20, 50, 100, 200]
    optimal_omegas = {}
    iterations_n = {}
    for n in ns:
        opt_omega = golden_section_search(sor_iterations, 1.0, 2.0, f_args=(n,))
        optimal_omegas[n] = opt_omega
        iterations_n[n] = sor_iterations(opt_omega, N=n)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(ns, [optimal_omegas[n] for n in ns], marker='o')
    ax1.set_title("Optimal $\\omega$ vs N", fontsize=20)
    ax1.set_xlabel("N", fontsize=16)
    ax1.set_ylabel("Optimal $\\omega$", fontsize=16)
    ax1.set_xticks(ns)
    ax2.plot(ns, [iterations_n[n] for n in ns], marker='o')
    ax2.set_title("Iterations to Converge vs N")
    ax2.set_xlabel("N", fontsize=16)
    ax2.set_ylabel("Iterations to Converge", fontsize=16)
    ax2.set_xticks(ns)
    plt.tight_layout()
    plt.savefig(f"{out}/sor_optimal_omega_and_iterations.png")
    plt.close()

    # Question K: Objects in domain — iteration sweeps and optimal omega
    N_k = 50
    epsilon = 1e-5
    tolerance_sizes = list(range(1, 11))   # 1 to 10
    num_rects_list = list(range(1, 20))    # 1 to 19

    # Sweep 1: objects_in_cntr (vary rectangle size)
    gauss_iterations_cntr = []
    sor_iterations_cntr = []
    sizes_cntr = []
    for size in tolerance_sizes:
        M = objects_in_cntr(N_k, size, size)
        gi, _, _ = gauss_seidel_iteration(N_k, epsilon=epsilon, M=M)
        si, _, _ = sor_iteration(N_k, omega=1.9, epsilon=epsilon, M=M)
        gauss_iterations_cntr.append(gi)
        sor_iterations_cntr.append(si)
        sizes_cntr.append(size)
    print("Question K: Sweep 1 (objects_in_cntr) done.")

    # Sweep 2: objects_in_domain_multiple (vary number of rectangles)
    gauss_iterations_multiple = []
    sor_iterations_multiple = []
    for num_rects in num_rects_list:
        rectangles = rectangles_for_count(N_k, num_rects)
        M = objects_in_domain_multiple(N_k, rectangles)
        gi, _, _ = gauss_seidel_iteration(N_k, epsilon=epsilon, M=M)
        si, _, _ = sor_iteration(N_k, omega=1.9, epsilon=epsilon, M=M)
        gauss_iterations_multiple.append(gi)
        sor_iterations_multiple.append(si)
    print("Question K: Sweep 2 (objects_in_domain_multiple) done.")

    plot_question_k_iteration_sweeps(
        sizes_cntr,
        gauss_iterations_cntr,
        sor_iterations_cntr,
        num_rects_list,
        gauss_iterations_multiple,
        sor_iterations_multiple,
        savepath=f"{out}/question_k_iteration_sweeps.png",
    )

    # Optimal omega sweep 1: objects_in_cntr
    optimal_omegas_cntr = []
    sizes_cntr_opt = []
    for size in tolerance_sizes:
        M = objects_in_cntr(N_k, size, size)
        def f_omega(omega):
            iters, _, _ = sor_iteration(N_k, omega=omega, epsilon=epsilon, M=M)
            return iters
        opt_omega = golden_section_search(f_omega, 1.7, 2.0)
        optimal_omegas_cntr.append(opt_omega)
        sizes_cntr_opt.append(size)
    print("Question K: Optimal omega sweep 1 done.")

    # Optimal omega sweep 2: objects_in_domain_multiple
    optimal_omegas_multiple = []
    num_rects_list_opt = []
    for num_rects in num_rects_list:
        rectangles = rectangles_for_count(N_k, num_rects)
        M = objects_in_domain_multiple(N_k, rectangles)
        def f_omega(omega):
            iters, _, _ = sor_iteration(N_k, omega=omega, epsilon=epsilon, M=M)
            return iters
        opt_omega = golden_section_search(f_omega, 1.7, 2.0)
        optimal_omegas_multiple.append(opt_omega)
        num_rects_list_opt.append(num_rects)
    print("Question K: Optimal omega sweep 2 done.")

    plot_question_k_optimal_omega_sweeps(
        sizes_cntr_opt,
        optimal_omegas_cntr,
        num_rects_list_opt,
        optimal_omegas_multiple,
        savepath=f"{out}/question_k_optimal_omega_sweeps.png",
    )


if __name__ == "__main__":
    main()
