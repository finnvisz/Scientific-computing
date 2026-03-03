import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# Row/col deltas for 4 directions: up, down, right, left
_DR = np.array([0, 0, 1, -1], dtype=np.int64)
_DC = np.array([1, -1, 0, 0], dtype=np.int64)


@njit
def monte_carlo_dla(grid: np.ndarray, target: int, p_s: float = 1) -> None:
    n = grid.shape[0]
    stuck = 0
    active_walker = False
    trapped = False
    walker_row, walker_col = 0, 0

    while stuck < target and not trapped:
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
                print(f"Walker is trapped at ({walker_row}, {walker_col})! This should never happen.")
                active_walker = False
                trapped = True
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


def make_seed(grid_size: int, seed_size: int) -> np.ndarray:
    grid = np.zeros((grid_size, grid_size), dtype=np.int64)
    mid = grid_size // 2
    start = mid - seed_size // 2
    for dr in range(-seed_size, 0):
        for dc in range(start, start + seed_size):
            grid[dr, dc] = 2
    return grid

grid_size = 100

result = monte_carlo_dla(make_seed(grid_size, 3), target=1000)
plt.imshow(result, cmap='viridis')
plt.title("DLA Cluster (p_s=1.0)")
plt.axis('off')
plt.show()

result = monte_carlo_dla(make_seed(grid_size, 3), target=1000, p_s=0.01)
plt.imshow(result, cmap='viridis')
plt.title("DLA Cluster (p_s=0.01)")
plt.axis('off')
plt.show()
