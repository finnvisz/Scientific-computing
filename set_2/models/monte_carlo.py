import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import njit
from tqdm import tqdm

# Row/col deltas for 4 directions: up, down, right, left
_DR = np.array([0, 0, 1, -1], dtype=np.int64)
_DC = np.array([1, -1, 0, 0], dtype=np.int64)


@njit
def monte_carlo_dla(grid: np.ndarray, target: int, p_s: float = 1) -> tuple:
    """Run DLA simulation.

    Returns
    -------
    final_grid      : (n, n) int64 — final state of the grid
    stick_positions : (target, 2) int16 — (row, col) of each stuck particle in order
    steps           : int — number of steps taken
    """
    grid = grid.copy()
    n = grid.shape[0]
    stuck = 0
    active_walker = False
    walker_row, walker_col = 0, 0
    steps = 0

    stick_positions = np.empty((target, 2), dtype=np.int16)

    while stuck < target:
        steps += 1
        if not active_walker:
            walker_row = 0
            walker_col = np.random.randint(0, n)
            while grid[walker_row, walker_col] != 0:
                walker_col = np.random.randint(0, n)
            active_walker = True
            continue

        # Propose a move: wrap columns, leave rows unbounded
        d = np.random.randint(0, 4)
        nr = walker_row + _DR[d]
        nc = (walker_col + _DC[d]) % n

        # Retry while proposed cell is occupied
        while 0 <= nr < n and grid[nr, nc] != 0:
            # If surrounded on all sides, abandon walker (should never happen)
            if ((walker_row == 0 or grid[walker_row - 1, walker_col]) and
                    (walker_row == n - 1 or grid[walker_row + 1, walker_col]) and
                    grid[walker_row, (walker_col - 1) % n] and grid[walker_row, (walker_col + 1) % n]):
                print("Walker trapped")
                active_walker = False
                break
            d = np.random.randint(0, 4)
            nr = walker_row + _DR[d]
            nc = (walker_col + _DC[d]) % n

        if not active_walker:
            continue

        # Reset if walker exits grid vertically
        if nr < 0 or nr >= n:
            active_walker = False
            continue

        walker_row, walker_col = nr, nc
    
        # Stick if adjacent to cluster
        if ((walker_row > 0 and grid[walker_row - 1, walker_col]) or
                grid[(walker_row + 1) % n, walker_col] or
                grid[walker_row, (walker_col - 1) % n] or
                grid[walker_row, (walker_col + 1) % n]):
            if np.random.random() < p_s:
                grid[walker_row, walker_col] = 1
                stick_positions[stuck, 0] = walker_row
                stick_positions[stuck, 1] = walker_col
                stuck += 1
                active_walker = False

    return grid, stick_positions, steps


def animate_dla(seed: np.ndarray, stick_positions: np.ndarray, duration: float = 10,
                title: str = "", filename: str = None) -> None:
    """Animate a DLA simulation, showing one frame per stick event.

    Parameters
    ----------
    seed        : initial grid passed to monte_carlo_dla
    stick_positions : (target, 2) int16 — (row, col) of each stuck particle in order
    duration    : target animation length in seconds
    filename    : if given, save to file instead of displaying
    """
    target = len(stick_positions)
    interval_ms = duration * 1000 / target

    def frame_gen():
        grid = seed.copy()
        for i in tqdm(range(target), desc=f"Generating {filename}"):
            r, c = stick_positions[i]
            grid[r, c] = 1
            yield grid.copy()

    fig, ax = plt.subplots()
    ax.axis('off')
    if title:
        ax.set_title(title)
    im = ax.imshow(seed, cmap='gray_r', vmin=0, vmax=2)

    def update(frame):
        im.set_data(frame)
        return (im,)

    anim = animation.FuncAnimation(fig, update, frames=frame_gen(),
                                   save_count=target, interval=interval_ms, blit=True)
    if filename:
        anim.save(filename)
    else:
        plt.show()
    plt.close()


def make_seed(grid_size: int, seed_size: int) -> np.ndarray:
    grid = np.zeros((grid_size, grid_size), dtype=np.int64)
    mid = grid_size // 2
    start = mid - seed_size // 2
    for dr in range(-seed_size, 0):
        for dc in range(start, start + seed_size):
            grid[dr, dc] = 2
    return grid

grid_size = 100
p_s_values = [1.0, 0.5, 0.2, 0.01]
dla_simulations = []
heatmaps = []

# Generate DLA simulations and animations
fig_dla, axes_dla = plt.subplots(2, 2, figsize=(10, 10))
fig_dla.suptitle("Monte Carlo DLA Simulations", fontsize=16)
axes_dla = axes_dla.flatten()

for idx, p_s in enumerate(p_s_values):
    seed = make_seed(grid_size, 3)
    sim, stick_positions, steps = monte_carlo_dla(seed, target=500, p_s=p_s)
    print(f"Monte Carlo DLA with p_s={p_s} took {steps} steps to reach 500 particles.")
    dla_simulations.append(sim)

    ax = axes_dla[idx]
    ax.imshow(sim, cmap='gray_r')
    ax.set_title(f"$p_s={p_s}$")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    animate_dla(seed, stick_positions, title=f"Monte Carlo DLA Animation, $p_s={p_s}$",
                filename=f"set_2/outputs/mc_dla_{str(p_s).replace('.', '_')}.gif")

fig_dla.tight_layout()
fig_dla.savefig("set_2/outputs/mc_dla.pdf", bbox_inches='tight')
plt.close(fig_dla)

# Generate heatmaps
fig_hm, axes_hm = plt.subplots(2, 2, figsize=(10, 10))
axes_hm = axes_hm.flatten()
fig_hm.suptitle("Monte Carlo DLA Heatmaps (n=20)", fontsize=16)

for idx, p_s in enumerate(p_s_values):
    heatmap = np.zeros((grid_size, grid_size), dtype=np.float64)
    for _ in tqdm(range(20), desc=f"Running DLA simulations (p_s={p_s})"):
        result, _, _ = monte_carlo_dla(make_seed(grid_size, 3), target=500, p_s=p_s)
        heatmap += (result > 0).astype(np.float64)
    heatmap /= 20
    heatmaps.append(heatmap)

    ax = axes_hm[idx]
    ax.imshow(heatmap, cmap='gray_r', vmin=0, vmax=1)
    ax.set_title(f"$p_s={p_s}$")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

fig_hm.tight_layout()
fig_hm.savefig("set_2/outputs/mc_heatmap.pdf", bbox_inches='tight')
plt.close(fig_hm)

