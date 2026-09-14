# Poland Energy Analytics

Analytical notebooks exploring the Polish power market: residual load, day-ahead
prices, solar capture factors, BESS economics, demand-temperature dynamics, and
renewable scarcity.

## Structure

- `notebooks/` — analysis notebooks, numbered in suggested reading order
  - `01_residual_load_and_prices.ipynb` — residual load vs. day-ahead prices
  - `02_solar_capture_factor.ipynb` — PV capture factor over time
  - `03_capture_factor_cannibalization.ipynb` — price cannibalization as PV grows
  - `04_bess_solar_economics.ipynb` — co-located solar + battery economics
  - `05_demand_temperature_ramps.ipynb` — demand sensitivity to temperature, ramps
  - `06_renewable_scarcity.ipynb` — scarcity events when wind + solar are low
- `sample_data/` — small demo dataset for running notebooks without external access
- `utils/` — shared plotting helpers and styling

## Quickstart

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
uv run jupyter lab
```

`uv sync` creates `.venv/`, installs dependencies from `pyproject.toml`, and
writes a lockfile (`uv.lock`) for reproducible environments.

Open any notebook in `notebooks/`. The notebooks default to **demo mode** and
read from `sample_data/pl_hourly_sample.csv` so they run end-to-end without
credentials.

## License

See [LICENSE](LICENSE).
