
## Requirements

To install Dependencies: pip install -r requirements.tx

## How to run

1. **Open a terminal and go to the repository** e.g.:

   ```bash
   cd /path/to/Scientific-computing/set_3
   ```

For each part of the assignment there is a seperate file, either python or notebook

## Running LBM.py

```bash
python LBM.py --N 25 --Re 100
```

- `--N`: grid resolution (default: 25)
- `--Re`: Reynolds number (default: 100)

Output is saved to `data/karman_vortex_street_n_{N}_re_{Re}.mp4`.

## Running WIFI.py

### Usage
- **Single Run:** 
```bash
python WIFI.py -x 2.3 -y 2.7 -res 0.01
```
a figure
- **Optimization:** 
```bash
python WIFI.py --optimize -step 0.1 -res 0.03
```
A csv and returns top 10 location
### Arguments
- `-x` : X-coordinate of the router (meters).
- `-y` : Y-coordinate of the router (meters).
- `-res` : Grid resolution (meters). Standard is `0.01` for 2.4 GHz.
- `-freq` : Frequency in GHz. Standard is `2.4`.
- `-step` : The spatial interval (meters) between candidate router locations during optimization.
- `--optimize` : If enabled, the script ignores the `-x` and `-y` flags and performs a global grid search across all legal locations in the apartment to find the maximum average signal strength.
### Outputs
- **Visualizations:** Saved in the `wififigs/` directory. 
- **Naming Convention:** `wifi_coverage_freq=<freq>_router=(<x>,<y>).png`
- **Data Results:** Optimization results are exported to `router_optimization_results_freq<freq>.csv` for further analysis. The top 10 is printed in the terminal.