import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
    
    for i in range(1, N):
        psi_now[i] = psi_past[i] + 0.5 * (c * dt *N)**2 * (psi_past[i+1] - 2*psi_past[i] + psi_past[i-1])
    
    # Time stepping
    results = [psi_past.copy()]
    for n in range(timesteps):
        psi_next = np.zeros_like(psi_now)
        for i in range(1, N):
            psi_next[i] = 2*psi_now[i] - psi_past[i] + (c * dt *N)**2 * (psi_now[i+1] - 2*psi_now[i] + psi_now[i-1])
        
        # Boundary conditions (Fixed ends)
        psi_next[0] = 0
        psi_next[N] = 0
        
        psi_past[:] = psi_now
        psi_now[:] = psi_next
        
        if n % 100 == 0: # Save every 100 steps for plotting
            results.append(psi_now.copy())
    return results

# Initial Conditions
psi_i = np.sin(2 * np.pi * x)
psi_ii = np.sin(5 * np.pi * x)
psi_iii = np.where((x > 0.2) & (x < 0.4), np.sin(5 * np.pi * x), 0)

# Run and Plot
initial_conditions = [psi_i, psi_ii, psi_iii]
titles = ["sin(2πx)", "sin(5πx)", " sin(5πx) if 1/5 < x < 2/5, else 0"]

plt.figure(figsize=(12, 8))
for j, initial_condition in enumerate(initial_conditions):
    result = simulate(initial_condition)
    plt.subplot(3, 1, j+1)
    for i, state in enumerate(result):
        plt.plot(x, state, alpha=0.5, label=f't={i*100*dt:.1f}')
    plt.title(titles[j])
    plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()