import numpy as np
import matplotlib.pyplot as plt

conv_threshold = 1e-5

def state_array(N, random_seed=None):
    M = np.zeros((N, N))
    if random_seed is not None:
        np.random.seed(random_seed)
        M += np.random.uniform(0, 1, size=(N, N))
    # boundary conditions
    M[0, :] = 1
    M[-1, :] = 0
    M[:, 0] = M[:, -1]
    return M

def plot_state_arrays(Ms, titles=[], suptitle=None, f=None):
    n = len(Ms)
    if len(titles) < n:
        raise ValueError("Not enough titles provided for the number of state arrays")
    plt.subplots(1, n, figsize=(5*n, 5))
    plt.suptitle(suptitle, fontsize=16)
    for i, M in enumerate(Ms):
        plt.subplot(1, n, i+1)
        plt.imshow(M, cmap='viridis')
        plt.colorbar()
        plt.title(titles[i])
    plt.tight_layout()
    if f is not None:
        plt.savefig(f)
    else:
        plt.show()
    plt.close()

def plot_error(errs, title=None, f=None):
    plt.title(title)
    plt.xlabel("k")
    plt.ylabel("$\delta$")
    plt.yscale("log")
    plt.plot(errs)
    if f is not None:
        plt.savefig(f)
    else:
        plt.show()
    plt.close()

def jacobi_step(M):
    new_M = np.copy(M)
    N = M.shape[0]
    for i in range(1, N-1):
        new_M[i, :] = 0.25 * (M[i-1, :] + M[i+1, :] + np.roll(M[i, :], 1) + np.roll(M[i, :], -1))
    return new_M

M_jacobi = state_array(50)
k = 0
errs_jacobi = []
while k < 10000:
    k += 1
    M2 = jacobi_step(M_jacobi)
    # check for convergence
    max_diff = np.max(np.abs(M2 - M_jacobi))
    errs_jacobi.append(max_diff)
    if np.max(np.abs(M2 - M_jacobi)) < conv_threshold:
        break
    M_jacobi = M2

print(f"Jacobi converged after {k} steps")
plot_error(errs_jacobi, title="Jacobi Error by iteration", f="jacobi_error.png")

def gauss_seidel_step(M):
    N = M.shape[0]
    for i in range(1, N-1):
        for j in range(N):
            M[i, j] = 0.25 * (M[i-1, j] + M[i+1, j] + M[i, j-1] + M[i, (j+1)%N])
    return M

M_gauss_seidel = state_array(50)
k = 0
errs_gauss_seidel = []
while k < 10000:
    k += 1
    M_old = np.copy(M_gauss_seidel)
    M_gauss_seidel = gauss_seidel_step(M_gauss_seidel)
    # check for convergence
    max_diff = np.max(np.abs(M_gauss_seidel - M_old))
    errs_gauss_seidel.append(max_diff)
    if max_diff < conv_threshold:
        break

print(f"Gauss-Seidel converged after {k} steps")
plot_error(errs_gauss_seidel, title="Gauss-Seidel Error by iteration", f="gauss_seidel_error.png")

def successive_over_relaxation_step(M, omega):
    N = M.shape[0]
    for i in range(1, N-1):
        for j in range(N):
            M[i, j] = (omega/4) * (M[i-1, j] + M[i+1, j] + M[i, j-1] + M[i, (j+1)%N]) + (1 - omega) * M[i, j]
    return M

def sor_iterations(omega, N=50, max_iter=10000, threshold=conv_threshold, random_seed=None):
    """Run SOR with a given omega and return the number of iterations to converge."""
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

errs = {}
omegas = [1.7, 1.85, 1.99]
for omega in omegas:
    M, k, errs[omega] = sor_iterations(omega, N=50, max_iter=10000, threshold=conv_threshold)
    print(f"Successive Over Relaxation ($\omega={omega}$) converged after {k} steps")


def golden_section_search(f, a, b, tol=1e-3, iter=0, max_iter=100, f_args=(), c=None, d=None, fc=None, fd=None):
    """Find the value of x in [a, b] that minimizes f(x) using recursive golden section search."""
    if iter >= max_iter:
        print("Maximum iterations reached in golden section search")
        return (a + b) / 2
    h = b - a
    if h < tol:
        print(f"Golden section search converged after {iter} iterations")
        return (a + b) / 2    
    gr = (1 + np.sqrt(5)) / 2  # golden ratio
    if c is None:
        c = a+ h * (1/(gr*gr))
    if d is None:
        d = a + h * (1/gr)
    if fc is None:
        fc = f(c, *f_args)[1]
    if fd is None:
        fd = f(d, *f_args)[1]
    print(f"Iter {iter+1}: a={a:.4f}, b={b:.4f}, c={c:.4f} (f={fc}), d={d:.4f} (f={fd})")
    if fc < fd:
        return golden_section_search(f, a, d, tol, iter+1, max_iter, f_args, d=c, fd=fc)
    else:
        return golden_section_search(f, c, b, tol, iter+1, max_iter, f_args, c=d, fc=fd)


gs_optimal_omega = golden_section_search(sor_iterations, 1.7, 2.0)
print(f"Golden Search Optimal omega: {gs_optimal_omega:.4f}")

M_sor, k, errs_sor = sor_iterations(gs_optimal_omega, N=50, max_iter=10000, threshold=conv_threshold)
print(f"Iterations at GS optimal omega: {k}")

plot_state_arrays([M_jacobi, M_gauss_seidel, M_sor], titles=["Jacobi", "Gauss-Seidel", "Successive Over Relaxation"], suptitle="Steady States by Method", f="steady_state_arrays.png")


plt.title(f"Convergence Rates")
plt.xlabel("k")
plt.ylabel("Error")
plt.plot(errs_jacobi, label="Jacobi")
plt.plot(errs_gauss_seidel, label="Gauss-Seidel")
plt.plot(errs_sor, label=f"SOR, $\omega={gs_optimal_omega:.4f}$ (optimal)")
for omega in omegas:
    plt.plot(errs[omega], label=f"SOR, $\omega={omega}$")
plt.yscale("log")
plt.axhline(y=conv_threshold, color='k', linestyle='--', label="Convergence Threshold")
plt.legend()
plt.savefig(f"convergence_rates.png")
plt.close()

M_r_sor, k, errs_r_sor = sor_iterations(gs_optimal_omega, N=50, max_iter=10000, threshold=conv_threshold, random_seed=42)
print(f"SOR with random initial state converged after {k} steps")
M_r_jacobi = state_array(50, random_seed=42)
M_r_gauss_seidel = state_array(50, random_seed=42)
errs_r_jacobi = []
errs_r_gauss_seidel = []
for _ in range(10000):
    M_r_j2 = jacobi_step(M_r_jacobi)
    errs_r_jacobi.append(np.max(np.abs(M_r_j2 - M_r_jacobi)))
    M_r_jacobi = M_r_j2
    if errs_r_jacobi[-1] < conv_threshold:
        break
print(f"Jacobi with random initial state converged after {len(errs_r_jacobi)} steps")
for _ in range(10000):
    M_gs_old = np.copy(M_r_gauss_seidel)
    gauss_seidel_step(M_r_gauss_seidel)
    errs_r_gauss_seidel.append(np.max(np.abs(M_r_gauss_seidel - M_gs_old)))
    if errs_r_gauss_seidel[-1] < conv_threshold:
        break
print(f"Gauss-Seidel with random initial state converged after {len(errs_r_gauss_seidel)} steps")

plt.title(f"Convergence Rates with Random Initial State")
plt.xlabel("k")
plt.ylabel("Error")
plt.plot(errs_r_jacobi, 'C0', label="Jacobi")
plt.plot(errs_r_gauss_seidel, 'C1', label="Gauss-Seidel")
plt.plot(errs_r_sor, 'C2', label=f"SOR, $\omega={gs_optimal_omega:.4f}$ (optimal)")
plt.plot(errs_jacobi, 'C0--')
plt.plot(errs_gauss_seidel, 'C1--')
plt.plot(errs_sor, 'C2--')
plt.yscale("log")
plt.axhline(y=conv_threshold, color='k', linestyle='--', label="Convergence Threshold")
plt.legend()
plt.savefig(f"convergence_rates_random.png")


ns = [5, 10, 20, 35, 50, 75, 100, 150, 200]
optimal_omegas = {}
iterations = {}
for n in ns:
    print(f"Finding optimal omega for N={n}...")
    opt_omega = golden_section_search(sor_iterations, 1.0, 2.0, f_args=(n,))
    optimal_omegas[n] = opt_omega
    iterations[n] = sor_iterations(opt_omega, N=n)[1]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(ns, [optimal_omegas[n] for n in ns], marker='o')
ax1.set_title("Optimal $\omega$ vs N", fontsize=16)
ax1.set_xlabel("N", fontsize=12)
ax1.set_ylabel("Optimal $\omega$", fontsize=12)
ax1.set_xticks(ns)
ax2.plot(ns, [iterations[n] for n in ns], marker='o')
ax2.set_title("Iterations to Converge vs N", fontsize=16)
ax2.set_xlabel("N", fontsize=12)
ax2.set_ylabel("Iterations to Converge", fontsize=12)
ax2.set_xticks(ns)
plt.tight_layout()
plt.savefig("sor_optimal_omega_and_iterations.png")