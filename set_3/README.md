
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
