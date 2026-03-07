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
    """
    grid = grid.copy()
    n = grid.shape[0]
    stuck = 0
    active_walker = False
    walker_row, walker_col = 0, 0

    stick_positions = np.empty((target, 2), dtype=np.int16)

    while stuck < target:
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

    return grid, stick_positions


def animate_dla(seed: np.ndarray, sim_results: tuple, duration: float = 10,
                title: str = "", filename: str = None) -> None:
    """Animate a DLA simulation, showing one frame per stick event.

    Parameters
    ----------
    seed        : initial grid passed to monte_carlo_dla
    sim_results : tuple returned by monte_carlo_dla
    duration    : target animation length in seconds
    filename    : if given, save to file instead of displaying
    """
    _, stick_positions = sim_results
    target = len(stick_positions)
    interval_ms = duration * 1000 / target

    def frame_gen():
        grid = seed.copy()
        for i in tqdm(range(target), desc="Generating animation"):
            r, c = stick_positions[i]
            grid[r, c] = 1
            yield grid.copy()

    fig, ax = plt.subplots()
    ax.axis('off')
    if title:
        ax.set_title(title)
    im = ax.imshow(seed, cmap='grey', vmin=0, vmax=2)

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


def run_heatmap(grid_size: int, seed_size: int, target: int, n_runs: int,
                p_s: float = 1.0, title: str = "", filename: str = None) -> None:
    """Run the DLA simulation n_runs times and plot a heatmap of particle frequency."""
    heatmap = np.zeros((grid_size, grid_size), dtype=np.float64)
    for _ in tqdm(range(n_runs), desc=f"Running DLA simulations (p_s={p_s})"):
        result, _ = monte_carlo_dla(make_seed(grid_size, seed_size), target=target, p_s=p_s)
        heatmap += (result > 0).astype(np.float64)
    heatmap /= n_runs

    plt.figure()
    plt.imshow(heatmap, cmap='hot', vmin=0, vmax=1)
    plt.colorbar().remove()
    plt.title(title or f"Monte Carlo DLA Heatmap (p_s={p_s}, n={n_runs})")
    plt.axis('off')
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()


def make_seed(grid_size: int, seed_size: int) -> np.ndarray:
    grid = np.zeros((grid_size, grid_size), dtype=np.int64)
    mid = grid_size // 2
    start = mid - seed_size // 2
    for dr in range(-seed_size, 0):
        for dc in range(start, start + seed_size):
            grid[dr, dc] = 2
    return grid

grid_size = 100

seed = make_seed(grid_size, 3)
sim = monte_carlo_dla(seed, target=500)
plt.imshow(sim[0], cmap='grey')
plt.title("Monte Carlo DLA Cluster $p_s=1.0$")
plt.axis('off')
plt.colorbar().remove()
plt.savefig("set_2/outputs/mc_dla_1_0.png", bbox_inches='tight')
animate_dla(seed, sim, title="Monte Carlo DLA Animation, $p_s=1.0$", filename="set_2/outputs/mc_dla_1_0.gif")

seed = make_seed(grid_size, 3)
sim = monte_carlo_dla(seed, target=500, p_s=0.3)
plt.imshow(sim[0], cmap='grey')
plt.title("Monte Carlo DLA Cluster $p_s=0.3$")
plt.axis('off')
plt.colorbar().remove()
plt.savefig("set_2/outputs/mc_dla_0_3.png", bbox_inches='tight')
animate_dla(seed, sim, title="Monte Carlo DLA Animation, $p_s=0.3$", filename="set_2/outputs/mc_dla_0_3.gif")

seed = make_seed(grid_size, 3)
sim = monte_carlo_dla(seed, target=500, p_s=0.01)
plt.imshow(sim[0], cmap='grey')
plt.title("Monte Carlo DLA Cluster $p_s=0.01$")
plt.axis('off')
plt.colorbar().remove()
plt.savefig("set_2/outputs/mc_dla_0_01.png", bbox_inches='tight')
animate_dla(seed, sim, title="Monte Carlo DLA Animation, $p_s=0.01$", filename="set_2/outputs/mc_dla_0_01.gif")

run_heatmap(grid_size, seed_size=3, target=500, n_runs=20, p_s=1.0, title="Monte Carlo DLA Heatmap, $p_s=1.0$", filename="set_2/outputs/mc_heatmap_1_0.png")
run_heatmap(grid_size, seed_size=3, target=500, n_runs=20, p_s=0.3, title="Monte Carlo DLA Heatmap, $p_s=0.3$", filename="set_2/outputs/mc_heatmap_0_3.png")
run_heatmap(grid_size, seed_size=3, target=500, n_runs=20, p_s=0.01, title="Monte Carlo DLA Heatmap, $p_s=0.01$", filename="set_2/outputs/mc_heatmap_0_01.png")

