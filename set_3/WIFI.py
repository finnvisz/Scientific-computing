import numpy as np
import yaml
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, linalg
from scipy.sparse.linalg import spsolve
import time
import pandas as pd
import csv

def draw_floorplan():
    """0 means air, 1 means wall"""
    with open('floorplan.yaml', 'r') as f:
        data = yaml.safe_load(f)['floorplan']

    plt.figure(figsize=(10, 8))

    # 2. Draw the Walls
    for wall in data['walls']:
        x_coords = [wall['start'][0], wall['end'][0]]
        y_coords = [wall['start'][1], wall['end'][1]]
        plt.plot(x_coords, y_coords, color='black', linewidth=3)

    # 3. Label the Rooms
    for room in data['rooms']:
        # Calculate center of the room for the label
        center_x = room['x'] + (room['width'] / 2)
        center_y = room['y'] + (room['height'] / 2)
        plt.text(center_x, center_y, room['name'].replace('_', ' ').title(), 
                ha='center', va='center', fontsize=9, fontweight='bold', color='blue')

    # Formatting the plot
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title("2D Floorplan Preview")
    plt.xlabel("Meters (X)")
    plt.ylabel("Meters (Y)")
    plt.show()

def create_floorplan(res):
    with open("floorplan.yaml") as f:
        data = yaml.safe_load(f)

    floorplan = data["floorplan"]
    outer_walls = floorplan["walls"]["outer_walls"]
    interior_walls = floorplan["walls"]["interior_walls"]

    # --- FIX 1: Calculate thickness in pixels based on resolution ---
    # If thickness is 0.15m and res is 0.05m, t should be 3 pixels, not 15.
    t = int(floorplan["wall_thickness"] / res)
    if t < 1: t = 1 # Ensure walls are at least 1 pixel wide

    width_m, height_m = 10.0, 8.0
    cols = int(width_m / res) 
    rows = int(height_m / res) 

    grid = np.full((rows, cols), "air", dtype=object)

    def draw_wall(start, end, inward=False):
        # --- FIX 2: Use float scaling to avoid rounding errors ---
        x1, y1 = int(start[0]/res), int(start[1]/res)
        x2, y2 = int(end[0]/res), int(end[1]/res)

        # vertical wall
        if x1 == x2:
            y_low, y_high = min(y1, y2), max(y1, y2)
            if inward:
                if x1 == 0:
                    grid[y_low:y_high, x1 : x1+t] = "wall"
                else:
                    grid[y_low:y_high, x1-t : x1] = "wall"
            else:
                grid[y_low:y_high, x1 - t//2 : x1 + (t - t//2)] = "wall"

        # horizontal wall
        elif y1 == y2:
            x_low, x_high = min(x1, x2), max(x1, x2)
            if inward:
                if y1 == 0:
                    grid[y1 : y1+t, x_low:x_high] = "wall"
                else:
                    grid[y1-t : y1, x_low:x_high] = "wall"
            else:
                grid[y1 - t//2 : y1 + (t - t//2), x_low:x_high] = "wall"

    for w in outer_walls:
        draw_wall(w["start"], w["end"], inward=True)
    for w in interior_walls:
        draw_wall(w["start"], w["end"], inward=False)

    return grid


def get_material_properties(grid, r, c, j=1j):
    """helper function, returns the refractive index, r and c in cm"""
    if grid[r][c] == 'wall':
        return 2.5 + 0.5 * j
    return 1

def get_source_term(x, y, x_router, y_router, A=10**4, sigma=0.2):
    return A * np.exp(-((x-x_router)**2+(y-y_router)**2)/(2*sigma**2))

def calculate_wavenumber(frequency=2.4*10**9):
    return 2 * np.pi * frequency / (3*10**8)

def solve_Helmholtz_equation(grid, res, x_router, y_router):
    h = res 
    rows, cols = grid.shape
    N = rows * cols
    k0 = calculate_wavenumber()
    
    A = lil_matrix((N, N), dtype=complex)
    f = np.zeros(N, dtype=complex)

    # print(f"Starting Matrix Assembly for {N} nodes...")
    # start_time = time.time()

    for r in range(rows):

        for c in range(cols):
            idx = r * cols + c
            
            if r == 0 or r == rows-1 or c == 0 or c == cols-1:
                A[idx, idx] = 1
                if r == 0: A[idx, idx + cols] = -1 / (1 + 1j * k0 * h)
                elif r == rows-1: A[idx, idx - cols] = -1 / (1 + 1j * k0 * h)
                elif c == 0: A[idx, idx + 1] = -1 / (1 + 1j * k0 * h)
                elif c == cols-1: A[idx, idx - 1] = -1 / (1 + 1j * k0 * h)
                continue

            n_complex = get_material_properties(grid, r, c)
            k_local = k0 * n_complex
            
            A[idx, idx] = (k_local**2) - (4 / h**2)
            A[idx, idx + 1] = 1 / h**2
            A[idx, idx - 1] = 1 / h**2
            A[idx, idx + cols] = 1 / h**2
            A[idx, idx - cols] = 1 / h**2
            
            f[idx] = get_source_term(c*h, r*h, x_router, y_router)

    A_csr = A.tocsr()

    # print("Solving with direct solver (spsolve)...")
    # solve_start = time.time()

    # This skips the 'iterations' and just calculates the answer
    u = spsolve(A_csr, f)

    # print(f"Solve complete! Time: {time.time() - solve_start:.2f}s")
    return u.reshape((rows, cols))

#########################
### Signal evaluation ###
#########################

def evaluate_room_strength(u_field, x_m, y_m, res, r_pixels=1):
    """
    Calculates average strength in a small region around (x_m, y_m)
    """
    magnitude = np.abs(u_field)
    # Normalize to 0dB peak
    signal_db = 20 * np.log10(magnitude / (np.max(magnitude) + 1e-12) + 1e-9)
    
    # Convert meters to indices
    r_idx = int(y_m / res)
    c_idx = int(x_m / res)
    
    # Define the small circular region (stencil)
    # We take a square slice and mask it, or just a small 3x3 average
    y_slice = slice(max(0, r_idx - r_pixels), min(u_field.shape[0], r_idx + r_pixels + 1))
    x_slice = slice(max(0, c_idx - r_pixels), min(u_field.shape[1], c_idx + r_pixels + 1))
    
    region = signal_db[y_slice, x_slice]
    return np.mean(region)

def evaluate_average_strength(u_field, res):
    """
    Sums the averages of the 4 specified locations.
    """
    locations = [
        ("Living Room", 1.0, 5.0),
        ("Kitchen",     2.0, 1.0),
        ("Bathroom",    9.0, 1.0),
        ("Bedroom 1",   9.0, 7.0)
    ]
    
    total_sum = 0
    for name, x, y in locations:
        avg = evaluate_room_strength(u_field, x, y, res)
        total_sum += avg
        
    print(f"{'Avg strength':15}: {total_sum/4:6.2f} dB")
    return total_sum / 4


#########################
### Plotting ############
#########################

def visualize_results(grid, u_field, router):
    magnitude = np.abs(u_field)
    signal_db = 20 * np.log10(magnitude + 1e-9)

    plt.figure(figsize=(9, 5))

    # Heatmap
    img = plt.imshow(signal_db, origin='lower', cmap='jet', extent=[0, 10, 0, 8])

    plt.title(f'freqency=2.4 GHz, resolution={res}',fontsize=10)
    plt.suptitle('Wifi coverage',fontsize=15)
    plt.colorbar(img, label='Signal Strength (dB)')

    # Walls overlay
    wall_mask = (grid == "wall").astype(float)
    plt.imshow(wall_mask, origin='lower', extent=[0, 10, 0, 8],
               alpha=0.3, cmap='Greys')
    
    avg_score = evaluate_average_strength(u_field, res)
    caption_text = f"Average Measured Signal Strength (Avg of 4 Points): {avg_score:.2f} dB"
    plt.figtext(0.5, 0.02, caption_text, wrap=True, horizontalalignment='center', 
                fontsize=12, bbox={'facecolor': 'orange', 'alpha': 0.2, 'pad': 5})

    plt.xlabel("Meters (X)")
    plt.ylabel("Meters (Y)")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"wififigs/wifi_coverage_router={router}.png")

#########################
### Optimization ########
#########################
def is_location_legal(x, y, grid, res, min_dist=0.5):
    """Checks if the router position is valid based on rules."""
    # Rule 1: Not inside a wall
    r_idx, c_idx = int(y / res), int(x / res)
    if grid[r_idx, c_idx] == 'wall':
        return False
    
    # Rule 2: At least 0.5m away from measurement points
    locations = [
        (1.0, 5.0), # Living Room
        (2.0, 1.0), # Kitchen
        (9.0, 1.0), # Bathroom
        (9.0, 7.0)  # Bedroom 1
    ]
    for mx, my in locations:
        dist = np.sqrt((x - mx)**2 + (y - my)**2)
        if dist < min_dist:
            return False
            
    return True

def optimization(grid, res, x_routers, y_routers):
    results = []
    total_spots = len(x_routers) * len(y_routers)
    
    print(f"Starting Grid Search: {total_spots} potential points...")
    start_time_all = time.time()
    results_file = "router_optimization_results.csv"
    with open(results_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x_m', 'y_m', 'avg_strength_db'])

    for x in x_routers:
        for y in y_routers:
            # ONLY solve if the location is legal
            if is_location_legal(x, y, grid, res):
                try:
                    u_field = solve_Helmholtz_equation(grid, res, x_router=x, y_router=y)
                    print('location router:', (x,y))
                    score = evaluate_average_strength(u_field, res)
                    results.append({
                        "X [m]": round(x, 2),
                        "Y [m]": round(y, 2),
                        "Average Strength [dB]": round(score, 2)
                    })
                    with open(results_file, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([round(x,2), round(y,2), round(score,2)])
                except Exception as e:
                    print(f"Solver failed at {x}, {y}: {e}")
            else:
                # Skip the heavy solver for illegal points
                pass 

    # Analysis and Output
    df = pd.DataFrame(results)
    top_positions = df.sort_values(by="Average Strength [dB]", ascending=False).head(10)
    
    print(f"\nOptimization complete in {(time.time() - start_time_all)/60:.2f} minutes.")
    print("\n--- TOP 10 LEGAL ROUTER LOCATIONS ---")
    print(top_positions.to_string(index=False))
    
    # Save results to CSV so you don't lose them
    df.to_csv("router_optimization_results.csv", index=False)

    return df



#########################
### Running #############
#########################

# test multiple locations

# res = 0.03
# offset = 0.015
# x_routers = np.arange(0.2, 9.9, 0.1)
# y_routers = np.arange(0.2, 7.9, 0.1)

# grid = create_floorplan(res)
# results_df = optimization(grid, res, x_routers, y_routers)

# test one location

res = 0.01
x_router = 3
y_router = 5
grid = create_floorplan(res)
u_field = solve_Helmholtz_equation(grid, res, x_router, y_router)
visualize_results(grid, u_field,(x_router,y_router))
print('avg strength:', evaluate_average_strength(u_field, res))


