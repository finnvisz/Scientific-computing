import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from tqdm import tqdm

# Row/col deltas for 4 directions: up, down, right, left
_DR = np.array([0, 0, 1, -1], dtype=np.int64)
_DC = np.array([1, -1, 0, 0], dtype=np.int64)


@njit
def monte_carlo_dla(grid: np.ndarray, target: int, p_s: float = 1) -> None:
    n = grid.shape[0]
    stuck = 0
    active_walker = False
    walker_row, walker_col = 0, 0

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
                print(f"Walker trapped at ({walker_row}, {walker_col})")
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
                stuck += 1
                active_walker = False
    return grid


def run_heatmap(grid_size: int, seed_size: int, target: int, n_runs: int,
                p_s: float = 1.0, title: str = "", filename: str = None) -> None:
    """Run the DLA simulation n_runs times and plot a heatmap of particle frequency."""
    heatmap = np.zeros((grid_size, grid_size), dtype=np.float64)
    for _ in tqdm(range(n_runs), desc=f"Running DLA simulations (p_s={p_s})"):
        result = monte_carlo_dla(make_seed(grid_size, seed_size), target=target, p_s=p_s)
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

result = monte_carlo_dla(make_seed(grid_size, 3), target=500)
plt.imshow(result, cmap='grey')
plt.title("Monte Carlo DLA Cluster $p_s=1.0$")
plt.axis('off')
plt.colorbar().remove()
plt.savefig("set_2/outputs/mc_dla_1_0.png", bbox_inches='tight')

result = monte_carlo_dla(make_seed(grid_size, 3), target=500, p_s=0.3)
plt.imshow(result, cmap='grey')
plt.title("Monte Carlo DLA Cluster $p_s=0.3$")
plt.axis('off')
plt.colorbar().remove()
plt.savefig("set_2/outputs/mc_dla_0_3.png", bbox_inches='tight')

result = monte_carlo_dla(make_seed(grid_size, 3), target=500, p_s=0.01)
plt.imshow(result, cmap='grey')
plt.title("Monte Carlo DLA Cluster $p_s=0.01$")
plt.axis('off')
plt.colorbar().remove()
plt.savefig("set_2/outputs/mc_dla_0_01.png", bbox_inches='tight')

run_heatmap(grid_size, seed_size=3, target=500, n_runs=20, p_s=1.0, title="Monte Carlo DLA Heatmap, $p_s=1.0$", filename="set_2/outputs/mc_heatmap_1_0.png")
run_heatmap(grid_size, seed_size=3, target=500, n_runs=20, p_s=0.3, title="Monte Carlo DLA Heatmap, $p_s=0.3$", filename="set_2/outputs/mc_heatmap_0_3.png")
run_heatmap(grid_size, seed_size=3, target=500, n_runs=20, p_s=0.01, title="Monte Carlo DLA Heatmap, $p_s=0.01$", filename="set_2/outputs/mc_heatmap_0_01.png")

