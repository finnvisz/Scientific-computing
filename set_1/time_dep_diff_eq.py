import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.special import erfc

# Parameters 
N = 50
D = 1.0
dx = 1.0 / N
# Stability: dt <= dx^2 / 4D
dt = 0.25 * (dx**2) / D 
r = (dt * D) / (dx**2)

# Initializing Grid 
c = np.zeros((N+1, N+1))
c[N, :] = 1.0  # Top boundary condition

def update(c_curr):
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

def analytic_solution(y, t, D, terms=100):
    if t == 0: return np.zeros_like(y)
    res = np.zeros_like(y)
    for i in range(terms):
        res += erfc((1 - y + 2*i) / (2 * np.sqrt(D * t))) - \
               erfc((1 + y + 2*i) / (2 * np.sqrt(D * t)))
    return res


#######################################
## E) Compare to analytical solution ##
#######################################

times_to_plot = [0, 0.001, 0.01, 0.1, 1.0]
curr_t = 0
step = 0
y_coords = np.linspace(0, 1, N + 1)
max_t = 1.01

plt.figure(figsize=(10, 6))

# Simulation Loop
while curr_t <= max_t:
    # Check if current time is one of our targets
    for t_target in times_to_plot:
        if np.isclose(curr_t, t_target, atol=dt/2):
            # Get numerical
            c_slice = c[:, N // 2]
            
            # Get Analytic Solution
            c_analytic = analytic_solution(y_coords, curr_t, D)
            
            # Plot both
            line, = plt.plot(y_coords, c_slice, 'o', markersize=4, label=f'Numerical t={curr_t:.3f}')
            plt.plot(y_coords, c_analytic, color=line.get_color(), linestyle='-', alpha=0.6, label=f'Analytic')

    c = update(c)
    curr_t += dt

# Formatting the plot
plt.title('Comparison of Numerical vs. Analytic Solutions')
plt.xlabel('y')
plt.ylabel('Concentration c(y)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f'set_1/figures/numerical_vs_analytic.png')

#########################
## F) 2D concentration ##
#########################

curr_t=0 # Reset current time
c = np.zeros((N+1, N+1))
c[N, :] = 1.0  # Reset Top boundary
# Loop until t=1.0
while curr_t <= 1.0:
    if any(np.isclose(curr_t, t_target, atol=dt/2) for t_target in times_to_plot):
        plt.imshow(c, extent=[0,1,0,1], origin='lower', cmap='viridis')
        plt.title(f"Time t = {curr_t:.3f}")
        plt.colorbar()
        plt.savefig(f'set_1/figures/diffusion_2D_t={curr_t:.3f}.png')
    c = update(c)
    curr_t += dt

##################
## G) Animation ##
##################

c_anim = np.zeros((N+1, N+1))
c_anim[N, :] = 1.0

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(c_anim, extent=[0, 1, 0, 1], origin='lower', cmap='viridis', animated=True)
ax.set_title("Evolution to Equilibrium")
fig.colorbar(im)

def animate(frame):
    global c_anim
    for _ in range(5): 
        c_anim = update(c_anim)
    im.set_array(c_anim)
    return [im]

ani = FuncAnimation(fig, animate, frames=200, interval=50, blit=True)
ani.save('set_1/figures/evolution_to_equilibrium.gif', writer='pillow', fps=30)
