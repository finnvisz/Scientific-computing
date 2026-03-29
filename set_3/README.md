
## Requirements

To install Dependencies: pip install -r requirements.tx

## How to run

1. **Open a terminal and go to the repository** e.g.:

   ```bash
   cd /path/to/Scientific-computing/set_3
   ```

For each part of the assignment there is a seperate file, either python or notebook

## Running FD.py (Finite Difference)
```bash
python FD.py
```
Parameters are set directly in the script:
- `nu`: kinematic viscosity (default: 0.001, giving Re_D ≈ 150)
- `nx`, `ny`: grid resolution (default: 881 × 165)
- `tend`: simulation end time (default: 6.0 s)
- `nit`: pressure Poisson sub-iterations (default: 50)

The time step `dt` is computed automatically from CFL and diffusive stability limits.

Output:
- `cfd_cylinder_fdm.gif` — velocity magnitude animation
- `cfd_cylinder_fdm_quiver.png` — final-state quiver plot
- `frames/fd_frame_XXXX.png` — 6 evenly-spaced snapshots for the report

Requires `numba` for JIT compilation of the solver kernels.

## Running FEM.ipynb (Finite Element)
Requires NGSolve and Netgen installed in your environment.

Run the cells sequentially in Jupyter:
```bash
jupyter notebook FEM.ipynb
```

Parameters (set in Cell 3):
- `nu`: kinematic viscosity (default: 0.00007, giving Re_D ≈ 2143)
- `tau`: time step (default: 0.0075 s)
- `tend`: simulation end time (default: 10 s)
- `maxh`: global mesh size (set in Cell 2, default: 0.025 m)

The notebook uses the NGSolve GUI for live visualisation (`netgen.gui` + `%gui tk`), so it must be run in a local Jupyter session, not in VS Code or JupyterLab headless mode.

Output:
- `cfd_animation.gif` — velocity magnitude animation
- `frames/fem_frame_XXXX.png` — 6 evenly-spaced snapshots for the report

## Running LBM.py

```bash
python LBM.py --N 25 --Re 100
```

- `--N`: grid resolution (default: 25)
- `--Re`: Reynolds number (default: 100)

Output is saved to `data/karman_vortex_street_n_{N}_re_{Re}.mp4`.

## Running WIFI.py

- **Single Run:** 
```bash
python WIFI.py -x 2.3 -y 2.7 -res 0.01
```

- **Optimization:** 
```bash
python WIFI.py --optimize -step 0.1 -res 0.03
```

Parameters:
- `-x` : X-coordinate of the router (meters).
- `-y` : Y-coordinate of the router (meters).
- `-res` : Grid resolution (meters). Standard is `0.01` for 2.4 GHz.
- `-freq` : Frequency in GHz. Standard is `2.4`.
- `-step` : The spatial interval (meters) between candidate router locations during optimization.
- `--optimize` : If enabled, the script ignores the `-x` and `-y` flags and performs a global grid search across all legal locations in the apartment to find the maximum average signal strength.

Outputs:
- **Visualizations:** Saved in the `wififigs/` directory. 
- **Naming Convention:** `wifi_coverage_freq=<freq>_router=(<x>,<y>).png`
- **Data Results:** Optimization results are exported to `router_optimization_results_freq<freq>.csv` for further analysis. The top 10 is printed in the terminal.