import numpy as np
import yaml
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, linalg
from scipy.sparse.linalg import spsolve
import time
import pandas as pd
import csv
import argparse
import sys


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

def solve_Helmholtz_equation(grid, res, freq, x_router, y_router):
    h = res 
    rows, cols = grid.shape
    N = rows * cols
    k0 = calculate_wavenumber(freq * 10 ** 9)
    
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

def visualize_results(grid, u_field, router, res, freq):
    magnitude = np.abs(u_field)
    signal_db = 20 * np.log10(magnitude + 1e-9)

    plt.figure(figsize=(9, 5))

    # Heatmap
    img = plt.imshow(signal_db, origin='lower', cmap='jet', extent=[0, 10, 0, 8])

    plt.title(f'freqency={freq} GHz, resolution={res}',fontsize=10)
    plt.suptitle('Wifi coverage',fontsize=15)
    plt.colorbar(img, label='Signal Strength (dB)')


    plt.scatter(router[0], router[1], marker='x', color='black', zorder=16, label = 'router')


    plt.scatter(5, 1, marker='o', s=50, color='black', 
                label='Measurement Points', zorder=8)

    # 2. Plot the rest without labels (so they don't repeat in the legend)
    other_points = [[1, 1], [2, 7], [9, 7]]
    for p in other_points:
        plt.scatter(p[0], p[1], marker='o', s=50, color='black', zorder=8)

    # 3. Add the 5cm circles (optional but good for the report)
    for p in [[5, 1], [1, 1], [2, 7], [9, 7]]:
        circle = plt.Circle((p[0], p[1]), 0.05, color='white', alpha=0.3, zorder=7)
        plt.gca().add_patch(circle)

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
    plt.legend()
    plt.savefig(f"wififigs/wifi_coverage_freq={freq}_router={router}.png")

#########################
### Optimization ########
#########################
def is_location_legal(x, y, grid, res, min_dist=0.5):
    """
    Checks if the router position is valid based on updated constraints:
    1. Not inside a wall (center-line check).
    2. At least 0.5m away from measurement points.
    3. 20cm buffer from 15cm-thick outer walls (Bounds: 0.35 to 9.65, 0.35 to 7.65).
    4. 20cm buffer from 15cm-thick interior walls (0.275m from center-line).
    """
    
    # --- Constraint 1: Outer Wall Thickness (15cm) + 20cm Buffer ---
    # Total offset from 0,0 and 10,8 is 0.35m
    if not (0.35 <= x <= 9.65 and 0.35 <= y <= 7.65):
        return False

    # --- Constraint 2: Not inside a wall (Grid-based check) ---
    r_idx, c_idx = int(y / res), int(x / res)
    # Bounds check to prevent IndexErrors near the edges
    if 0 <= r_idx < grid.shape[0] and 0 <= c_idx < grid.shape[1]:
        if grid[r_idx, c_idx] == 'wall':
            return False
    else:
        return False # Outside grid is illegal

    # --- Constraint 3: Interior Wall Surface Buffer (20cm) ---
    interior_walls = [
        ([0, 3], [3, 3]), ([4, 3], [6, 3]), ([7, 3], [10, 3]), # y=3 segments
        ([2.5, 0], [2.5, 2]), ([6, 3], [6, 8]),               # Verticals
        ([7, 0], [7, 1.5]), ([7, 2.5], [7, 3])                # Bathroom Door
    ]
    
    for start, end in interior_walls:
        # Distance from point (x,y) to line segment
        p1, p2 = np.array(start), np.array(end)
        p3 = np.array([x, y])
        # Vector logic for segment distance
        d = p2 - p1
        if np.linalg.norm(d) > 0:
            t = np.clip(np.dot(p3 - p1, d) / np.dot(d, d), 0, 1)
            projection = p1 + t * d
            dist_to_int_wall = np.linalg.norm(p3 - projection)
            if dist_to_int_wall < 0.275:
                return False

    # --- Constraint 4: At least 0.5m away from measurement points ---
    meas_locations = [(5.0, 1.0), (1.0, 1.0), (2.0, 7.0), (9.0, 7.0)]
    for mx, my in meas_locations:
        dist = np.sqrt((x - mx)**2 + (y - my)**2)
        if dist < min_dist:
            return False
            
    return True

def optimization(grid, res, freq, x_routers, y_routers):
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
                    u_field = solve_Helmholtz_equation(grid, res, freq, x_router=x, y_router=y)
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
    print(f"\n--- TOP 10 LEGAL ROUTER LOCATIONS FREQUENCY {freq} ---")
    print(top_positions.to_string(index=False))
    
    # Save results to CSV so you don't lose them
    df.to_csv(f"router_optimization_results_freq{freq}.csv", index=False)

    return df



#########################
### Running #############
#########################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WiFi Helmholtz Simulation & Optimization")

    # Core Parameters
    parser.add_argument("-res", type=float, default=0.01, help="Grid resolution (m)")
    parser.add_argument("-freq", type=float, default=2.4, help="Frequency (GHz)")
    
    # Single Run Coordinates
    parser.add_argument("-x", type=float, default=2.3, help="Single run Router X")
    parser.add_argument("-y", type=float, default=2.7, help="Single run Router Y")

    # Optimization Toggle
    parser.add_argument("--optimize", action="store_true", help="Run full grid optimization")
    parser.add_argument("-step", type=float, default=0.1, help="Optimization grid step (m)")

    args = parser.parse_args()
    grid = create_floorplan(args.res)

    if args.optimize:
        print(f"--- STARTING OPTIMIZATION (Step={args.step}m, Res={args.res}m) ---")
        # Define ranges for the whole 10x8 apartment
        x_range = np.arange(0.2, 10, args.step)
        y_range = np.arange(0.2, 8, args.step)
        
        # Call your existing optimization function
        optimization(grid, args.res, args.freq, x_range, y_range)
    
    else:
        print(f"--- RUNNING SINGLE POSITION ({args.x}, {args.y}) ---")
        if is_location_legal(args.x, args.y, grid, args.res):
            u_field = solve_Helmholtz_equation(grid, args.res, args.freq, args.x, args.y)
            visualize_results(grid, u_field, (args.x, args.y), args.res, args.freq)
        else:
            print(f"Location ({args.x}, {args.y}) is ILLEGAL.")

