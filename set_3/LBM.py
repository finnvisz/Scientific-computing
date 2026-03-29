"""
2D Lattice Boltzmann Method (LBM) solver for the Navier-Stokes equations.
Simulates a Karman vortex street behind a cylinder.

Lattice: D2Q9
Collision: BGK (single relaxation time), upgradeable to TRT
Inlet: Zou-He velocity BC (parabolic profile)
Outlet: Zou-He pressure BC (rho = 1)
Walls: Simple bounce-back (top/bottom)
Cylinder: Interpolated bounce-back (Bouzidi et al.)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit, prange
from tqdm import tqdm
import argparse

# Given physical parameters:
L_phys = 2.2        # domain length [m]
H_phys = 0.41       # domain height [m] (0.15 + 0.1 + 0.16)
D_phys = 0.1        # cylinder diameter [m]
cx_phys = 0.15      # cylinder center x [m] (from left wall)
cy_phys = 0.20      # cylinder center y [m] (0.15m from bottom + 0.05m radius)
U_phys = 1.0       # max inlet velocity [m/s] (sets time scale)

# 9 discrete velocities for 2D lattice Boltzmann.
# Index ordering: 0=rest, 1=E, 2=N, 3=W, 4=S, 5=NE, 6=NW, 7=SW, 8=SE
#
#   6 2 5
#    \|/
#   3-0-1
#    /|\
#   7 4 8

# Lattice velocity vectors (c_i), shape (9, 2): each row is (cx, cy)
c = np.array([
    [0,  0],   # 0: rest
    [1,  0],   # 1: E
    [0,  1],   # 2: N
    [-1, 0],   # 3: W
    [0, -1],   # 4: S
    [1,  1],   # 5: NE
    [-1, 1],   # 6: NW
    [-1,-1],   # 7: SW
    [1, -1],   # 8: SE
])

# Lattice weights
w = np.array([
    4/9,                            # rest
    1/9, 1/9, 1/9, 1/9,            # cardinal
    1/36, 1/36, 1/36, 1/36,        # diagonal
])

# Opposite direction indices: for bounce-back, f_i bounces to f_opp[i]
opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

# Speed of sound squared (for D2Q9: cs^2 = 1/3)
cs2 = 1.0 / 3.0

# Reshaped for broadcasting with (Nx, Ny) fields
c9x = c[:, 0].reshape(9, 1, 1)   # shape (9, 1, 1)
c9y = c[:, 1].reshape(9, 1, 1)   # shape (9, 1, 1)
w9  = w.reshape(9, 1, 1)

class Grid:
    """Stores all lattice parameters, grid coordinates, and obstacle mask."""

    def __init__(self, Re, N, u_lb=0.08):
        """
        Parameters:
            Re:     Reynolds number (Re = U * D / nu)
            N:      lattice nodes across cylinder diameter (resolution control)
            u_lb:   max inlet velocity in lattice units (0.04-0.1 for stability)
            U_phys: physical max inlet velocity [m/s] (sets the time scale)
        """
        self.Re = Re
        self.N = N
        self.u_lb = u_lb
        self.U_phys = U_phys

        # Lattice spacing
        self.dx = D_phys / N

        # Grid dimensions
        self.Nx = int(L_phys / self.dx)
        self.Ny = int(H_phys / self.dx) + 1 # +1 to account for the integer division truncation

        # Physical time step: u_lb = U_phys * dt / dx  →  dt = u_lb * dx / U_phys
        self.dt = u_lb * self.dx / U_phys

        # Cylinder center and radius in lattice units
        self.cx_lat = cx_phys / self.dx
        self.cy_lat = cy_phys / self.dx
        self.R_lat = N / 2

        # Kinematic viscosity and relaxation time
        # STABILITY CONSTRAINT: tau > 0.5 (ideally > 0.51)
        self.nu_lb = u_lb * N / Re
        self.tau = 3 * self.nu_lb + 0.5

        # Grid coordinates and obstacle mask
        x = np.arange(self.Nx)
        y = np.arange(self.Ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        self.solid = (X - self.cx_lat)**2 + (Y - self.cy_lat)**2 <= self.R_lat**2

        # Precompute solid node indices for fast zeroing
        si, sj = np.where(self.solid)
        self.solid_i = si.astype(np.int64)
        self.solid_j = sj.astype(np.int64)

        # Precompute boundary links for the cylinder
        self.boundary_links = self._compute_boundary_links()

        print(f"Grid: {self.Nx} x {self.Ny}, tau = {self.tau:.4f}, "
              f"nu_lb = {self.nu_lb:.6f}, Ma = {u_lb / (1/3)**0.5:.3f}, "
              f"dx = {self.dx:.4e} m, dt = {self.dt:.4e} s, "
              f"boundary links = {self.boundary_links[0].shape[0]}")

    def _compute_boundary_links(self):
        """
        For each fluid node adjacent to the cylinder, find lattice directions
        that point into the solid and compute the fractional wall distance q.

        Returns list of (i, j, k, q) tuples, where:
            (i, j) = fluid node
            k      = direction index pointing toward the solid
            q      = fractional distance to the cylinder surface along c[k]
                     (q in [0, 1], where 0 = wall at the fluid node,
                      1 = wall at the solid node)
        """
        links = []
        for i in tqdm(range(self.Nx), desc="Computing cylinder boundary links"):
            for j in range(self.Ny):
                if self.solid[i, j]:
                    continue  # skip solid nodes

                for k in range(1, 9):  # skip 0 (rest direction)
                    ni = i + c[k, 0]
                    nj = j + c[k, 1]

                    # Check if neighbor is in bounds and solid
                    if (0 <= ni < self.Nx and 0 <= nj < self.Ny
                            and self.solid[ni, nj]):

                        # Ray-circle intersection to find q
                        # Ray: P + t * d, where P = (i, j), d = c[k]
                        # Circle: |X - C|^2 = R^2
                        oc_x = i - self.cx_lat
                        oc_y = j - self.cy_lat
                        dx = c[k, 0]
                        dy = c[k, 1]

                        a = dx**2 + dy**2
                        b = 2 * (oc_x * dx + oc_y * dy)
                        cc = oc_x**2 + oc_y**2 - self.R_lat**2

                        discriminant = b**2 - 4 * a * cc
                        if discriminant < 0:
                            print(f"WARNING: negative discriminant at "
                                  f"({i},{j}) dir {k}, falling back to q=0.5")
                            q = 0.5
                        else:
                            # Smallest positive t gives the nearest intersection
                            sqrt_disc = np.sqrt(discriminant)
                            t1 = (-b - sqrt_disc) / (2 * a)
                            t2 = (-b + sqrt_disc) / (2 * a)
                            # t should be in (0, 1] — pick the smallest positive
                            if t1 > 0:
                                q = t1
                            else:
                                q = t2
                            q = np.clip(q, 1e-4, 1.0)

                        links.append((i, j, k, q))

        bl_i = np.array([l[0] for l in links], dtype=np.int64)
        bl_j = np.array([l[1] for l in links], dtype=np.int64)
        bl_k = np.array([l[2] for l in links], dtype=np.int64)
        bl_q = np.array([l[3] for l in links], dtype=np.float64)
        return bl_i, bl_j, bl_k, bl_q

@njit
def equilibrium(rho, ux, uy):
    """
    Compute the equilibrium distribution f_eq for all 9 directions.

    Parameters:
        rho: density field, shape (Nx, Ny)
        ux:  x-velocity field, shape (Nx, Ny)
        uy:  y-velocity field, shape (Nx, Ny)

    Returns:
        feq: equilibrium distributions, shape (9, Nx, Ny)
    """
    cu = c9x * ux + c9y * uy              # (9, Nx, Ny)
    usq = ux**2 + uy**2                   # (Nx, Ny)
    feq = w9 * rho * (1 + cu/cs2 + cu**2/(2*cs2**2) - usq/(2*cs2))
    return feq

@njit(parallel=True)
def compute_macroscopic(f, rho, ux, uy):
    """
    Compute macroscopic density and velocity from distribution functions.
    Writes results into pre-allocated arrays to avoid allocation overhead.

    Parameters:
        f:   distribution functions, shape (9, Nx, Ny)
        rho: output density, shape (Nx, Ny)
        ux:  output x-velocity, shape (Nx, Ny)
        uy:  output y-velocity, shape (Nx, Ny)
    """
    Nx = f.shape[1]
    Ny = f.shape[2]
    for i in prange(Nx):
        for j in range(Ny):
            r = 0.0
            mx = 0.0
            my = 0.0
            for k in range(9):
                fk = f[k, i, j]
                r += fk
                mx += fk * c[k, 0]
                my += fk * c[k, 1]
            rho[i, j] = r
            if r > 0:
                ux[i, j] = mx / r
                uy[i, j] = my / r
            else:
                ux[i, j] = 0.0
                uy[i, j] = 0.0

@njit(parallel=True)
def collide_inplace(f, rho, ux, uy, tau):
    """
    BGK collision operator (in-place).
    Modifies f directly to avoid allocating a new array.

    Parameters:
        f:   distributions, shape (9, Nx, Ny) — modified in-place
        rho: density, shape (Nx, Ny)
        ux:  x-velocity, shape (Nx, Ny)
        uy:  y-velocity, shape (Nx, Ny)
        tau: relaxation time (scalar)
    """
    feq = equilibrium(rho, ux, uy)
    inv_tau = 1.0 / tau
    for k in prange(9):
        for i in range(f.shape[1]):
            for j in range(f.shape[2]):
                f[k, i, j] -= inv_tau * (f[k, i, j] - feq[k, i, j])

@njit(parallel=True)
def stream(f, f_out):
    """
    Stream distributions to neighboring nodes.
    Writes into pre-allocated f_out buffer to avoid allocation overhead.

    Parameters:
        f:     post-collision distributions, shape (9, Nx, Ny)
        f_out: pre-allocated output buffer, shape (9, Nx, Ny)
    """
    Nx, Ny = f.shape[1], f.shape[2]
    for k in prange(9):
        cx = c[k, 0]
        cy = c[k, 1]
        for i in range(Nx):
            ni = (i + cx) % Nx
            for j in range(Ny):
                nj = (j + cy) % Ny
                f_out[k, ni, nj] = f[k, i, j]

@njit
def apply_wall_bc(f, f_prev):
    """
    No-slip bounce-back on top (j = Ny-1) and bottom (j = 0) walls.

    After streaming, distributions that came from outside the domain are
    garbage (np.roll wrapped them). Replace with the opposite-direction
    post-collision values from before streaming.

    Parameters:
        f:      post-streaming distributions, shape (9, Nx, Ny)
        f_prev: post-collision distributions (pre-streaming), shape (9, Nx, Ny)
    """
    # Bottom wall (j=0): outgoing were 4,7,8 (downward), bounce to 2,5,6
    f[2, :, 0] = f_prev[4, :, 0]
    f[5, :, 0] = f_prev[7, :, 0]
    f[6, :, 0] = f_prev[8, :, 0]
    # Top wall (j=-1): outgoing were 2,5,6 (upward), bounce to 4,7,8
    f[4, :, -1] = f_prev[2, :, -1]
    f[7, :, -1] = f_prev[5, :, -1]
    f[8, :, -1] = f_prev[6, :, -1]


@njit
def apply_inlet_bc(f, ux_inlet_profile):
    """
    Zou-He velocity boundary condition at the left wall (i = 0).
    Prescribes a parabolic velocity profile ux(y) with uy = 0.

    After streaming, directions 1, 5, 8 are unknown (pointing into domain
    from the left). The known directions at i=0 are: 0, 2, 3, 4, 6, 7.

    Parameters:
        f: distribution functions, shape (9, Nx, Ny)
        ux_inlet_profile: prescribed ux at inlet, shape (Ny,)
    """
    ux = ux_inlet_profile  # shape (Ny,)

    rho = (1 / (1 - ux)) * (f[0, 0, :] + f[2, 0, :] + f[4, 0, :]
                             + 2 * (f[3, 0, :] + f[6, 0, :] + f[7, 0, :]))

    f[1, 0, :] = f[3, 0, :] + (2/3) * rho * ux
    f[5, 0, :] = f[7, 0, :] + (1/6) * rho * ux + 0.5 * (f[4, 0, :] - f[2, 0, :])
    f[8, 0, :] = f[6, 0, :] + (1/6) * rho * ux - 0.5 * (f[4, 0, :] - f[2, 0, :])

@njit
def apply_outlet_bc(f):
    """
    Zou-He pressure boundary condition at the right wall (i = Nx-1).
    Prescribes rho = 1.0 (reference density) with uy = 0.

    After streaming, directions 3, 6, 7 are unknown (pointing into domain
    from the right). The known directions at i=-1 are: 0, 1, 2, 4, 5, 8.

    Parameters:
        f: distribution functions, shape (9, Nx, Ny)
    """
    rho_out = 1.0

    ux = -1 + (1/rho_out) * (f[0, -1, :] + f[2, -1, :] + f[4, -1, :]
                              + 2 * (f[1, -1, :] + f[5, -1, :] + f[8, -1, :]))

    f[3, -1, :] = f[1, -1, :] - (2/3) * rho_out * ux
    f[7, -1, :] = f[5, -1, :] - (1/6) * rho_out * ux + 0.5 * (f[2, -1, :] - f[4, -1, :])
    f[6, -1, :] = f[8, -1, :] - (1/6) * rho_out * ux - 0.5 * (f[2, -1, :] - f[4, -1, :])

@njit(parallel=True)
def apply_cylinder_bc(f, f_prev, bl_i, bl_j, bl_k, bl_q):
    """
    Bouzidi interpolated bounce-back for the cylinder.

    Parameters:
        f, f_prev:        post-streaming and post-collision distributions, shape (9, Nx, Ny)
        bl_i, bl_j, bl_k: fluid node coordinates and direction index, shape (n,)
        bl_q:             fractional wall distance, shape (n,)
    """
    for idx in prange(bl_i.shape[0]):
        i = bl_i[idx]
        j = bl_j[idx]
        k = bl_k[idx]
        q = bl_q[idx]
        k_opp = opp[k]

        if q < 0.5:
            ii = i - c[k, 0]
            jj = j - c[k, 1]
            f[k_opp, i, j] = (2*q * f_prev[k, i, j]
                              + (1 - 2*q) * f_prev[k, ii, jj])
        else:
            f[k_opp, i, j] = (1/(2*q) * f_prev[k, i, j]
                              + (1 - 1/(2*q)) * f_prev[k_opp, i, j])

@njit
def zero_solid(f, solid_i, solid_j):
    """
    Zero out distributions at solid nodes using precomputed index arrays.

    Parameters:
        f:                 distributions, shape (9, Nx, Ny)
        solid_i, solid_j:  coordinate arrays of solid nodes
    """
    for idx in range(solid_i.shape[0]):
        i = solid_i[idx]
        j = solid_j[idx]
        for k in range(9):
            f[k, i, j] = 0.0

def parabolic_profile(Ny, u_max):
    """
    Generate a parabolic (Poiseuille) velocity profile for the inlet.

    Parameters:
        Ny:    number of lattice nodes in y
        u_max: maximum velocity (in lattice units)

    Returns:
        profile: velocity at each y node, shape (Ny,)
    """
    H = Ny - 1  # distance between walls in lattice units
    y = np.arange(Ny)
    ux = u_max * 4 * y * (H - y) / H**2
    return ux

def run(grid, t_ms, fps=30, plot=True, plot_warmup=False, animation_file=None):
    """
    Main simulation loop.

    For each time step:
        1. Compute macroscopic quantities (rho, ux, uy) from f
        2. Collision: f_post = collide(f, rho, ux, uy, tau)
        3. Save f_post (needed for Bouzidi interpolation)
        4. Streaming: f = stream(f_post)
        5. Apply boundary conditions (order matters):
           a. Zou-He inlet
           b. Zou-He outlet
           c. Wall bounce-back (top/bottom)
           d. Cylinder interpolated bounce-back
        6. Set f = 0 inside solid nodes (safety measure)
        7. Periodically visualize / save results

    Parameters:
        t_ms:          physical simulation time [ms]
        fps:           target frames per second for visualization/animation
        plot_warmup:   whether to visualize during warmup phase
        plot:          whether to visualize the flow (vorticity and speed)
        animation_file: if set, store frames and save animation to this file
                        instead of real-time rendering (requires ffmpeg for .mp4
                        or Pillow for .gif)
    """
    dt = grid.dt
    num_steps = int(round(t_ms * 1e-3 / dt))
    warmup_steps = grid.Nx
    plot_every = max(1, int(round(1.0 / (dt * fps))))
    print(f"Simulation: {num_steps} steps ({t_ms} ms), "
          f"warmup: {warmup_steps} steps ({warmup_steps * dt * 1e3:.1f} ms), "
          f"plot every {plot_every} steps ({plot_every * dt * 1e3:.2f} ms), "
          f"{int(num_steps / plot_every + 1)} frames @ {fps} fps")
    profile = parabolic_profile(grid.Ny, grid.u_lb)
    # Initialize at rest — avoids acoustic shock from sudden cylinder obstruction
    f = equilibrium(np.ones((grid.Nx, grid.Ny)), np.zeros((grid.Nx, grid.Ny)), np.zeros((grid.Nx, grid.Ny)))
    initial_noise = 1e-6 * (np.random.rand(*f.shape) - 0.5)  # small random noise to trigger vortex shedding
    f += initial_noise  # add small noise to trigger vortex shedding
    f[:, grid.solid] = 0
    bl_i, bl_j, bl_k, bl_q = grid.boundary_links

    f_postcol = np.empty_like(f)
    rho = np.empty((grid.Nx, grid.Ny))
    ux = np.empty((grid.Nx, grid.Ny))
    uy = np.empty((grid.Nx, grid.Ny))

    fig, axes = None, None
    frames = [] if animation_file else None
    step = 0
    should_plot = plot and not animation_file
    for i in tqdm(range(num_steps + warmup_steps + 1), desc="Running LBM simulation"): # one extra step to capture final frame
        step = i
        try:
            # Ramp inlet velocity linearly to avoid initial acoustic shock
            ramp = min(step / warmup_steps, 1.0) if warmup_steps > 0 else 1.0

            compute_macroscopic(f, rho, ux, uy)
            collide_inplace(f, rho, ux, uy, grid.tau)
            f_postcol[:] = f
            stream(f_postcol, f)
            apply_inlet_bc(f, profile * ramp)
            apply_outlet_bc(f)
            apply_wall_bc(f, f_postcol)
            apply_cylinder_bc(f, f_postcol, bl_i, bl_j, bl_k, bl_q)
            zero_solid(f, grid.solid_i, grid.solid_j)
            if (plot_warmup or step >= warmup_steps) and (step - (warmup_steps if not plot_warmup else 0)) % plot_every == 0:
                compute_macroscopic(f, rho, ux, uy)
                if animation_file:
                    t_phys = (step - warmup_steps) * grid.dt
                    frames.append((ux.copy(), uy.copy(), t_phys))
                elif should_plot:
                    t_phys = (step - warmup_steps) * grid.dt
                    fig, axes = plot_flow(ux, uy, grid, t_phys, fig, axes)
        except FloatingPointError:
            print(f"Numerical instability at step {step}, stopping simulation.")
            break

    if animation_file:
        print(f"Rendering animation with {len(frames)} frames...")
        save_animation(frames, grid, animation_file, fps=fps)
    elif should_plot:
        t_phys = (step - warmup_steps) * grid.dt
        plot_flow(ux, uy, grid, t_phys, fig, axes)
        plt.ioff()
        plt.show()

def save_animation(frames, grid, filename, fps=60):
    """
    Render stored frames into an animation and save to file.

    Parameters:
        frames:   list of (ux, uy, step) tuples
        grid:     Grid object (for solid mask and color scale limits)
        filename: output file path (.mp4, .gif, etc.)
        fps:      frames per second in the output animation
    """
    dx = grid.dx
    vel_scale = dx / grid.dt  # lattice velocity → m/s
    vort_scale = 1.0 / grid.dt  # lattice vorticity → 1/s
    extent = [0, grid.Nx * dx, 0, grid.Ny * dx]  # [x_min, x_max, y_min, y_max] in meters

    vlim = grid.u_lb * vort_scale * 0.5
    smax = grid.U_phys * 2.0

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica'],
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
    })
    fig, axes = plt.subplots(2, 1, figsize=(10, 4), gridspec_kw={'hspace': 0.35})

    ux0, uy0, t0 = frames[0]
    vorticity = (np.gradient(uy0, axis=0) - np.gradient(ux0, axis=1)) * vort_scale
    speed = np.sqrt(ux0**2 + uy0**2) * vel_scale
    vorticity = np.where(grid.solid, np.nan, vorticity)
    speed = np.where(grid.solid, np.nan, speed)

    im_vort = axes[0].imshow(vorticity.T, origin='lower', cmap='RdBu_r',
                             vmin=-vlim, vmax=vlim, aspect='equal', interpolation='bilinear',
                             extent=extent)
    fig.colorbar(im_vort, ax=axes[0], label='Vorticity (1/s)', shrink=0.8, pad=0.02)
    axes[0].set_ylabel('y (m)')
    title = axes[0].set_title(f'Vorticity  |  t = {t0:.4f} s', fontweight='bold')

    im_speed = axes[1].imshow(speed.T, origin='lower', cmap='viridis',
                              vmin=0, vmax=smax, aspect='equal', interpolation='bilinear',
                              extent=extent)
    fig.colorbar(im_speed, ax=axes[1], label='|u| (m/s)', shrink=0.8, pad=0.02)
    axes[1].set_xlabel('x (m)')
    axes[1].set_ylabel('y (m)')
    axes[1].set_title('Velocity magnitude', fontweight='bold')

    def update(frame_idx):
        ux_f, uy_f, t_f = frames[frame_idx]
        vort = (np.gradient(uy_f, axis=0) - np.gradient(ux_f, axis=1)) * vort_scale
        spd = np.sqrt(ux_f**2 + uy_f**2) * vel_scale
        vort = np.where(grid.solid, np.nan, vort)
        spd = np.where(grid.solid, np.nan, spd)
        im_vort.set_data(vort.T)
        im_speed.set_data(spd.T)
        title.set_text(f'Vorticity  |  t = {t_f:.4f} s')
        return im_vort, im_speed, title

    with tqdm(total=len(frames), desc="Rendering animation", unit="frame", leave=False) as pbar:
        def update_tracked(frame_idx):
            result = update(frame_idx)
            pbar.update(1)
            return result
        anim = FuncAnimation(fig, update_tracked, frames=len(frames), blit=True, interval=1000//fps)
        anim.save(filename, fps=fps)
    plt.close(fig)
    print(f"Animation saved to {filename}")


def plot_flow(ux, uy, grid, t_phys, fig=None, axes=None):
    """
    Two-panel visualization: vorticity (top) and velocity magnitude (bottom).

    Parameters:
        ux, uy:  velocity fields in lattice units, shape (Nx, Ny)
        grid:    Grid object (for solid mask and unit conversion)
        t_phys:  current physical time [s]
        fig, axes: existing figure/axes for live updating (None to create new)

    Returns:
        fig, axes:  for reuse in subsequent calls
    """
    dx = grid.dx
    vel_scale = dx / grid.dt
    vort_scale = 1.0 / grid.dt
    extent = [0, grid.Nx * dx, 0, grid.Ny * dx]

    # Compute derived fields in physical units
    vorticity = (np.gradient(uy, axis=0) - np.gradient(ux, axis=1)) * vort_scale
    speed = np.sqrt(ux**2 + uy**2) * vel_scale

    # Mask solid nodes with NaN for clean rendering
    vorticity = np.where(grid.solid, np.nan, vorticity)
    speed = np.where(grid.solid, np.nan, speed)

    if fig is None:
        vlim = grid.u_lb * vort_scale * 0.5
        smax = grid.U_phys * 2.0

        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica'],
            'font.size': 8,
            'axes.labelsize': 9,
            'axes.titlesize': 10,
        })
        fig, axes = plt.subplots(2, 1, figsize=(10, 4),
                                 gridspec_kw={'hspace': 0.35})
        axes[0].im = axes[0].imshow(
            vorticity.T, origin='lower', cmap='RdBu_r',
            vmin=-vlim, vmax=vlim, aspect='equal', interpolation='bilinear',
            extent=extent, animated=True)
        fig.colorbar(axes[0].im, ax=axes[0], label='Vorticity (1/s)',
                     shrink=0.8, pad=0.02)
        axes[0].set_ylabel('y (m)')
        axes[0].title_text = axes[0].set_title(f'Vorticity  |  t = {t_phys:.4f} s', fontweight='bold')
        axes[0].title_text.set_animated(True)

        axes[1].im = axes[1].imshow(
            speed.T, origin='lower', cmap='viridis',
            vmin=0, vmax=smax, aspect='equal', interpolation='bilinear',
            extent=extent, animated=True)
        fig.colorbar(axes[1].im, ax=axes[1], label='|u| (m/s)',
                     shrink=0.8, pad=0.02)
        axes[1].set_xlabel('x (m)')
        axes[1].set_ylabel('y (m)')
        axes[1].set_title('Velocity magnitude', fontweight='bold')

        plt.ion()
        fig.show()
        fig.canvas.draw()
        fig._bg = fig.canvas.copy_from_bbox(fig.bbox)
    else:
        axes[0].im.set_data(vorticity.T)
        axes[0].title_text.set_text(f'Vorticity  |  t = {t_phys:.4f} s')

        axes[1].im.set_data(speed.T)

        fig.canvas.restore_region(fig._bg)
        axes[0].draw_artist(axes[0].im)
        axes[0].draw_artist(axes[0].title_text)
        axes[1].draw_artist(axes[1].im)
        fig.canvas.blit(fig.bbox)
        fig.canvas.flush_events()

    return fig, axes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--Re", type=float, default=25)
    args = parser.parse_args()

    np.seterr(all='raise')  # raise exceptions on numerical issues (e.g. NaN, inf)
    Re = args.Re
    N = args.N
    u_lb = 0.1
    grid = Grid(Re, N, u_lb)
    animation_file = f'set_3/data/karman_vortex_street_n_{N}_re_{int(Re)}.mp4'
    run(grid, t_ms=10000, plot=True, plot_warmup=False, animation_file=animation_file)
