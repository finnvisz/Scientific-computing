import numpy as np
import matplotlib.pyplot as plt

conv_threshold = 1e-5

def state_array(N):
    M = np.zeros((N, N))
    # all 1s where y=1
    M[0, :] = 1
    return M

def plot_state_array(M, title=None, f=None):
    plt.imshow(M, cmap='gray')
    plt.colorbar()
    if title is not None:
        plt.title(title)
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

M = state_array(50)
k = 0
errs = []
while k < 10000:
    k += 1
    M2 = jacobi_step(M)
    # check for convergence
    max_diff = np.max(np.abs(M2 - M))
    errs.append(max_diff)
    if np.max(np.abs(M2 - M)) < conv_threshold:
        break
    M = M2

print(f"Jacobi converged after {k} steps")
plot_state_array(M, title="Jacobi", f="jacobi.png")
plot_error(errs, title="Jacobi Error by iteration", f="jacobi_error.png")

def gauss_seidel_step(M):
    N = M.shape[0]
    for i in range(1, N-1):
        for j in range(N):
            M[i, j] = 0.25 * (M[i-1, j] + M[i+1, j] + M[i, j-1] + M[i, (j+1)%N])
    return M

M = state_array(50)
k = 0
errs = []
while k < 10000:
    k += 1
    M_old = np.copy(M)
    M = gauss_seidel_step(M)
    # check for convergence
    max_diff = np.max(np.abs(M - M_old))
    errs.append(max_diff)
    if max_diff < conv_threshold:
        break

print(f"Gauss-Seidel converged after {k} steps")
plot_state_array(M, title="Gauss-Seidel", f="gauss_seidel.png")
plot_error(errs, title="Gauss-Seidel Error by iteration", f="gauss_seidel_error.png")

def successive_over_relaxation_step(M, omega):
    N = M.shape[0]
    for i in range(1, N-1):
        for j in range(N):
            M[i, j] = (omega/4) * (M[i-1, j] + M[i+1, j] + M[i, j-1] + M[i, (j+1)%N]) + (1 - omega) * M[i, j]
    return M

errs = {}
omegas = [1.7, 1.8, 1.9, 1.99]
for omega in omegas:
    M = state_array(50)
    k = 0
    errs[omega] = []
    while k < 10000:
        k += 1
        M_old = np.copy(M)
        M = successive_over_relaxation_step(M, omega)
        # check for convergence
        max_diff = np.max(np.abs(M - M_old))
        errs[omega].append(max_diff)
        if max_diff < conv_threshold:
            break

    print(f"Successive Over Relaxation ($\omega={omega}$) converged after {k} steps")
    plot_state_array(M, title=f"Successive Over Relaxation, $\omega={omega}$", f=f"successive_over_relaxation_{omega}.png")

def sor_iterations(omega, N=50, max_iter=10000, threshold=conv_threshold):
    """Run SOR with a given omega and return the number of iterations to converge."""
    M = state_array(N)
    for k in range(1, max_iter + 1):
        M_old = np.copy(M)
        M = successive_over_relaxation_step(M, omega)
        if np.max(np.abs(M - M_old)) < threshold:
            return k
    return max_iter

def golden_section_search(f, a, b, tol=1e-3, iter=0, max_iter=100, f_args=()):
    """Find the value of x in [a, b] that minimizes f(x) using recursive golden section search."""
    if iter >= max_iter:
        print("Maximum iterations reached in golden section search")
        return (a + b) / 2
    if b - a < tol:
        print(f"Golden section search converged after {iter} iterations")
        return (a + b) / 2
    gr = (1 + np.sqrt(5)) / 2  # golden ratio
    x1 = b - (b - a) / gr
    x2 = a + (b - a) / gr
    f1 = f(x1, *f_args)
    f2 = f(x2, *f_args)
    if f1 < f2:
        return golden_section_search(f, a, x2, tol, iter + 1, max_iter, f_args)
    else:
        return golden_section_search(f, x1, b, tol, iter + 1, max_iter, f_args)


gs_optimal_omega = golden_section_search(sor_iterations, 1.7, 2.0)
print(f"Golden Search Optimal omega: {gs_optimal_omega:.4f}")
print(f"Iterations at GS optimal omega: {sor_iterations(gs_optimal_omega)}")

gs_omega = gs_optimal_omega

M = state_array(50)
k = 0
gs_errs = []
for k in range(1, 10000):
    M_old = np.copy(M)
    M = successive_over_relaxation_step(M, gs_omega)
    # check for convergence
    max_diff = np.max(np.abs(M - M_old))
    gs_errs.append(max_diff)
    if max_diff < conv_threshold:
        break

print(f"Successive Over Relaxation ($\omega={gs_omega:.4f}$) converged after {k} steps")
plot_state_array(M, title=f"Successive Over Relaxation, Optimal $\omega={gs_omega:.4f}$", f=f"successive_over_relaxation_optimal.png")

plt.title(f"Successive Over Relaxation Error")
plt.xlabel("k")
plt.ylabel("Error")
for omega in omegas:
    plt.plot(errs[omega], label=f"$\omega={omega}$")
plt.plot(gs_errs, label=f"$\omega={gs_omega:.4f}$ (optimal)", linestyle='--')
plt.yscale("log")
plt.legend()
plt.savefig(f"successive_over_relaxation_error.png")
plt.close()

ns = [5, 20, 50, 100, 200]
optimal_omegas = {}
iterations = {}
for n in ns:
    opt_omega = golden_section_search(sor_iterations, 1.7, 2.0, f_args=(n,))
    optimal_omegas[n] = opt_omega
    iterations[n] = sor_iterations(opt_omega, N=n)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(ns, [optimal_omegas[n] for n in ns], marker='o')
ax1.set_title("Optimal $\omega$ vs N")
ax1.set_xlabel("N")
ax1.set_ylabel("Optimal $\omega$")
ax1.set_xticks(ns)
ax2.plot(ns, [iterations[n] for n in ns], marker='o')
ax2.set_title("Iterations to Converge vs N")
ax2.set_xlabel("N")
ax2.set_ylabel("Iterations to Converge")
ax2.set_xticks(ns)
plt.tight_layout()
plt.savefig("sor_optimal_omega_and_iterations.png")