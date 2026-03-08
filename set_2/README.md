# Set 2 — DLA and Gray-Scott Reaction-Diffusion

Code for **Diffusion-Limited Aggregation (DLA)** and the **Gray-Scott reaction-diffusion model**.

## Requirements

- Python 3
- NumPy, SciPy, Matplotlib

## How to run

1. **Open a terminal and go to the repository root** (the folder that contains `set_2`), e.g.:

   ```bash
   cd /path/to/Scientific-computing
   ```

   You must be in `Scientific-computing`, not inside `set_2`.

2. **Run the main script as a module:**

   ```bash
   python3 -m set_2.main
   ```

   If you see `ModuleNotFoundError: No module named 'set_2'`, you are not in the repo root—move up one level so that `set_2` is a subfolder of your current directory, then run the command again.

## What it does

- **2.1 DLA:** Builds DLA clusters for several values of η and compares SOR convergence for different ω.
- **2.3 Gray-Scott:** Runs the Gray-Scott model for different (f, k) and plots U and V concentration profiles at chosen time steps.

## Where figures are saved

Figures are written into the **set_2/outputs** folder. When you run from the repo root, they appear in `Scientific-computing/set_2/outputs`, for example:

- `set_2/outputs/DLA_eta_comparison_omega=1.7.png`
- `set_2/outputs/result_stats_omega.png`
- `set_2/outputs/gray_scott_concentration_profiles_Default Parameters.png`
- `set_2/outputs/gray_scott_concentration_profiles_Random Parameters.png`
- `set_2/outputs/gray_scott_concentration_profiles_Coral Growth Parameters.png`

## Layout

```
set_2/
├── README.md
├── main.py
├── __init__.py
├── models/
│   ├── DLA.py             
│   ├── Gray_Scott.py       
│   └── monte_carlo.py      
└── outputs
```
