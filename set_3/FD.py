import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import njit
 
# DOMAIN & GRID

Lx = 2.2
Ly = 0.41
nx = 881
ny = 165
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
 
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)
 

# CYLINDER MASK

cx, cy, cr = 0.2, 0.2, 0.05
dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
 
solid = dist <= cr
halo_radius = cr + max(dx, dy) * 1.1
solid_expanded = dist <= halo_radius

# PHYSICAL PARAMETERS

nu   = 0.001        # kinematic viscosity  (Re_D ≈ 150)
rho  = 1.0
tend = 6.0
nit  = 50
 
H = Ly
u_inlet_full = 1.5 * 4.0 * y * (H - y) / H**2
U_max = u_inlet_full.max()
t_ramp = 1.0
 
# Auto-compute safe dt from both stability limits
dt_diff = 0.20 * min(dx, dy)**2 / nu       # diffusive (safety factor 0.20)
dt_CFL  = 0.40 * min(dx, dy) / (U_max + 1e-12)  # CFL (safety factor 0.40)
dt = min(dt_diff, dt_CFL)
 
Re_cell = U_max * dx / nu
print(f"Grid: {nx}x{ny},  dx={dx:.5f}, dy={dy:.5f}")
print(f"dt = {dt:.2e}  (auto)  |  CFL limit ~ {dt_CFL:.2e}  |  Diff limit ~ {dt_diff:.2e}")
print(f"Cell Reynolds number: Re_cell = {Re_cell:.2f}  (upwind handles any value)")
print(f"Global Re_D = {U_max * 2*cr / nu:.0f}")

# FIELD INITIALISATION

u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
 

# NJIT-COMPILED FUNCTIONS

@njit
def build_up_b(b, rho, dt, dx, dy, u, v, solid_exp):
    """Pressure Poisson RHS."""
    ny, nx = u.shape
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            if solid_exp[i, j]:
                b[i, j] = 0.0
            else:
                dudx = (u[i, j+1] - u[i, j-1]) / (2.0 * dx)
                dvdy = (v[i+1, j] - v[i-1, j]) / (2.0 * dy)
                dudy = (u[i+1, j] - u[i-1, j]) / (2.0 * dy)
                dvdx = (v[i, j+1] - v[i, j-1]) / (2.0 * dx)
                b[i, j] = rho * (
                    (1.0 / dt) * (dudx + dvdy)
                    - dudx * dudx
                    - 2.0 * dudy * dvdx
                    - dvdy * dvdy
                )
    return b
 
 
@njit
def pressure_poisson(p, b, dx, dy, nit, solid):
    """Jacobi iteration for pressure."""
    ny, nx = p.shape
    dx2 = dx * dx
    dy2 = dy * dy
    denom = 2.0 * (dx2 + dy2)
    coeff = dx2 * dy2 / denom
    pn = np.empty_like(p)
 
    for _ in range(nit):
        for i in range(ny):
            for j in range(nx):
                pn[i, j] = p[i, j]
 
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                p[i, j] = (
                    (pn[i, j+1] + pn[i, j-1]) * dy2 +
                    (pn[i+1, j] + pn[i-1, j]) * dx2
                ) / denom - coeff * b[i, j]
 
        for j in range(nx):
            p[0, j]    = p[1, j]
            p[ny-1, j] = p[ny-2, j]
        for i in range(ny):
            p[i, 0] = p[i, 1]
        for i in range(ny):
            p[i, nx-1] = 0.0
        for i in range(ny):
            for j in range(nx):
                if solid[i, j]:
                    p[i, j] = 0.0
    return p
 
 
@njit
def momentum_step_upwind(u, v, un, vn, p, dt, dx, dy, rho, nu,
                         u_inlet, solid_exp):
    """
    Forward-Euler momentum with FIRST-ORDER UPWIND convection.
 
    Upwind rule for du/dx:
        if u[i,j] >= 0:  du/dx = (u[i,j] - u[i,j-1]) / dx   (backward)
        if u[i,j] <  0:  du/dx = (u[i,j+1] - u[i,j]) / dx   (forward)
 
    This is unconditionally stable w.r.t. cell Reynolds number.
    """
    ny, nx = u.shape
    dt_dx2 = dt / (dx * dx)
    dt_dy2 = dt / (dy * dy)
    dt_2rhodx = dt / (2.0 * rho * dx)
    dt_2rhody = dt / (2.0 * rho * dy)
 
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            ui = un[i, j]
            vi = vn[i, j]
 
            # --- Upwind convection for u ---
            if ui >= 0.0:
                dudx = (un[i, j] - un[i, j-1]) / dx
            else:
                dudx = (un[i, j+1] - un[i, j]) / dx
 
            if vi >= 0.0:
                dudy = (un[i, j] - un[i-1, j]) / dy
            else:
                dudy = (un[i+1, j] - un[i, j]) / dy
 
            # --- Upwind convection for v ---
            if ui >= 0.0:
                dvdx = (vn[i, j] - vn[i, j-1]) / dx
            else:
                dvdx = (vn[i, j+1] - vn[i, j]) / dx
 
            if vi >= 0.0:
                dvdy = (vn[i, j] - vn[i-1, j]) / dy
            else:
                dvdy = (vn[i+1, j] - vn[i, j]) / dy
 
            # Diffusion (central, 2nd order)
            u_diff = (
                dt_dx2 * (un[i, j+1] - 2.0*un[i, j] + un[i, j-1]) +
                dt_dy2 * (un[i+1, j] - 2.0*un[i, j] + un[i-1, j])
            )
            v_diff = (
                dt_dx2 * (vn[i, j+1] - 2.0*vn[i, j] + vn[i, j-1]) +
                dt_dy2 * (vn[i+1, j] - 2.0*vn[i, j] + vn[i-1, j])
            )
 
            # Pressure gradient (central, 2nd order)
            dpdx = dt_2rhodx * (p[i, j+1] - p[i, j-1])
            dpdy = dt_2rhody * (p[i+1, j] - p[i-1, j])
 
            # Update
            u[i, j] = ui - dt * (ui * dudx + vi * dudy) - dpdx + nu * u_diff
            v[i, j] = vi - dt * (ui * dvdx + vi * dvdy) - dpdy + nu * v_diff
 
    # --- Boundary conditions ---
    for j in range(nx):
        u[0, j] = 0.0;    v[0, j] = 0.0
        u[ny-1, j] = 0.0; v[ny-1, j] = 0.0
 
    for i in range(ny):
        u[i, 0] = u_inlet[i]
        v[i, 0] = 0.0
 
    for i in range(ny):
        u[i, nx-1] = u[i, nx-2]
        v[i, nx-1] = v[i, nx-2]
 
    for i in range(ny):
        for j in range(nx):
            if solid_exp[i, j]:
                u[i, j] = 0.0
                v[i, j] = 0.0
 
    return u, v
 
 
# WARM-UP

print("\nCompiling Numba functions ...")
b_tmp = np.zeros((ny, nx))
build_up_b(b_tmp, rho, dt, dx, dy, u, v, solid_expanded)
pressure_poisson(p, b_tmp, dx, dy, 1, solid)
un_tmp = u.copy(); vn_tmp = v.copy()
momentum_step_upwind(u, v, un_tmp, vn_tmp, p, dt, dx, dy, rho, nu,
                     u_inlet_full * 0.0, solid_expanded)
u[:] = 0.0; v[:] = 0.0; p[:] = 0.0
print("Done.\n")
 

# TIME LOOP

nt = int(tend / dt)
print_every = max(1, nt // 200)
save_every  = max(1, nt // 300)
 
vel_history  = []
time_history = []
b = np.zeros((ny, nx))
 
print(f"Starting simulation: {nt} steps, tend={tend}, dt={dt:.2e}")
print(f"Saving every {save_every} steps, printing every {print_every} steps")
print(f"Inflow ramp over {t_ramp} s\n")
 
for n in range(nt):
    un = u.copy()
    vn = v.copy()
 
    t_curr = (n + 1) * dt
    ramp = min(1.0, t_curr / t_ramp)
    u_inlet = u_inlet_full * ramp
 
    build_up_b(b, rho, dt, dx, dy, un, vn, solid_expanded)
    pressure_poisson(p, b, dx, dy, nit, solid)
    momentum_step_upwind(u, v, un, vn, p, dt, dx, dy, rho, nu,
                         u_inlet, solid_expanded)
 
    if (n + 1) % print_every == 0:
        u_max_now = np.nanmax(np.abs(u))
        v_max_now = np.nanmax(np.abs(v))
 
        if np.isnan(u_max_now) or np.isnan(v_max_now) or u_max_now > 1e6:
            print(f"Step {n+1}/{nt}  t={t_curr:.4f}  *** DIVERGED ***")
            break
 
        div_max = np.max(np.abs(
            (u[1:-1, 2:] - u[1:-1, 0:-2]) / (2*dx) +
            (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2*dy)
        ))
        print(
            f"Step {n+1:>7d}/{nt}  t={t_curr:7.4f}  "
            f"|u|_max={u_max_now:.4f}  |v|_max={v_max_now:.4f}  "
            f"|div|_max={div_max:.3e}"
        )
 
    if (n + 1) % save_every == 0:
        vel_mag = np.sqrt(u**2 + v**2)
        vel_history.append(vel_mag.copy())
        time_history.append(t_curr)
 
print(f"\nSimulation complete.  Saved {len(vel_history)} frames.\n")
 

# ANIMATION

if len(vel_history) > 0:
    print("Generating GIF animation ...")
 
    fig, ax = plt.subplots(figsize=(14, 3), dpi=120)
    fig.subplots_adjust(bottom=0.18, top=0.85)
 
    vmax = min(max(np.max(f) for f in vel_history), 2.5)
 
    im = ax.pcolormesh(X, Y, vel_history[0], shading='gouraud',
                       cmap='jet', vmin=0, vmax=vmax)
    fig.colorbar(im, ax=ax, label='|u| [m/s]', shrink=0.9)
 
    theta = np.linspace(0, 2*np.pi, 100)
    cyl_x = cx + cr * np.cos(theta)
    cyl_y = cy + cr * np.sin(theta)
 
    def animate(i):
        ax.clear()
        ax.pcolormesh(X, Y, vel_history[i], shading='gouraud',
                      cmap='jet', vmin=0, vmax=vmax)
        ax.fill(cyl_x, cyl_y, color='grey', zorder=5)
        ax.plot(cyl_x, cyl_y, 'k-', linewidth=0.8, zorder=6)
        ax.set_aspect('equal')
        ax.set_title(f'Velocity magnitude  —  t = {time_history[i]:.3f} s')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
 
    ani = animation.FuncAnimation(fig, animate, frames=len(vel_history),
                                  interval=80, blit=False)
    gif_path = 'cfd_cylinder_fdm.gif'
    ani.save(gif_path, writer='pillow', fps=15)
    plt.close(fig)
    print(f"Animation saved to {gif_path}")
 
    fig2, ax2 = plt.subplots(figsize=(14, 3), dpi=120)
    skip = 6
    ax2.quiver(X[::skip, ::skip], Y[::skip, ::skip],
               u[::skip, ::skip], v[::skip, ::skip], scale=30)
    ax2.fill(cyl_x, cyl_y, color='lightgrey', zorder=5)
    ax2.plot(cyl_x, cyl_y, 'k-', linewidth=0.8, zorder=6)
    ax2.set_aspect('equal')
    ax2.set_title(f'Velocity vectors at t = {nt*dt:.3f} s')
    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('y [m]')
    fig2.tight_layout()
    fig2.savefig('cfd_cylinder_fdm_quiver.png', dpi=150)
    plt.close(fig2)
    print("Quiver plot saved to cfd_cylinder_fdm_quiver.png")


    from PIL import Image as PILImage
    import os

    os.makedirs("frames", exist_ok=True)
    gif = PILImage.open(gif_path)
    total = gif.n_frames
    n_extract = 6
    indices = [int(i * (total - 1) / (n_extract - 1)) for i in range(n_extract)]

    for idx in indices:
        gif.seek(idx)
        frame = gif.convert("RGB")
        frame.save(f"frames/fd_frame_{idx:04d}.png", dpi=(150, 150))
        print(f"  Saved frames/fd_frame_{idx:04d}.png")

    print(f"Extracted {n_extract} frames to frames/")
else:
    print("No frames saved — try increasing tend.")
