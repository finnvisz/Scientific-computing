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

def jacobi_step(M):
    new_M = np.copy(M)
    N = M.shape[0]
    for i in range(1, N-1):
        new_M[i, :] = 0.25 * (M[i-1, :] + M[i+1, :] + np.roll(M[i, :], 1) + np.roll(M[i, :], -1))
    return new_M

M = state_array(50)
k = 0
while True:
    k += 1
    M2 = jacobi_step(M)
    # check for convergence
    if np.max(np.abs(M2 - M)) < conv_threshold:
        break
    M = M2

print(f"Jacobi converged after {k} steps")
plot_state_array(M, title="Jacobi", f="jacobi.png")


def gauss_seidel_step(M):
    N = M.shape[0]
    for i in range(1, N-1):
        for j in range(N):
            M[i, j] = 0.25 * (M[i-1, j] + M[i+1, j] + M[i, j-1] + M[i, (j+1)%N])
    return M

M = state_array(50)
k = 0
while True:
    k += 1
    M_old = np.copy(M)
    M = gauss_seidel_step(M)
    # check for convergence
    max_diff = np.max(np.abs(M - M_old))
    if max_diff < conv_threshold:
        break

print(f"Gauss-Seidel converged after {k} steps")
plot_state_array(M, title="Gauss-Seidel", f="gauss_seidel.png")

def successive_over_relaxation_step(M, omega):
    N = M.shape[0]
    for i in range(1, N-1):
        for j in range(N):
            M[i, j] = (omega/4) * (M[i-1, j] + M[i+1, j] + M[i, j-1] + M[i, (j+1)%N]) + (1 - omega) * M[i, j]
    return M


errs = {}
omegas = [1.7, 1.85, 1.99]
for omega in omegas:
    M = state_array(50)
    k = 0
    errs[omega] = []
    while True:
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

plt.title(f"Successive Over Relaxation Error")
plt.xlabel("k")
plt.ylabel("Error")
for omega in omegas:
    plt.plot(errs[omega], label=f"$\omega={omega}$")
plt.yscale("log")
plt.legend()
plt.savefig(f"successive_over_relaxation_error.png")
plt.close()

# find optimal omega w/ binary search
def binary_search(min, max, N, errs={}, tol=1e-5):
    omega = (min + max) / 2
    M = state_array(N)
    k = 0
    errs[omega] = []
    while True:
        k += 1
        M_old = np.copy(M)
        M = successive_over_relaxation_step(M, omega)
        # check for convergence
        max_diff = np.max(np.abs(M - M_old))
        errs[omega].append(max_diff)
        if max_diff < tol:
            break
    