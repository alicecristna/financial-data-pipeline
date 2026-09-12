# Synthetic demo data

The CSVs in this directory are **not real market data** and carry no economic
meaning. They are deterministic synthetic series used so that the offline
tests and `demo.py` do not depend on any downloaded snapshot. Asset names are
neutral on purpose.

## How the files were generated

Generator: `scripts/generate_demo_data.py` (committed). Reproduce with:

```bash
python scripts/generate_demo_data.py
```

Model, for each series independently:

- `SEED = 20240603`, `N_DAYS = 500`, business days starting `2024-06-03`.
- Per-series RNG: `numpy.random.default_rng([SEED, index])`, where `index` is
  the position in the generator's `SPECS` mapping (starting at 0).
- Daily simple returns: `r_t ~ Normal(drift, vol)`.
- Close: `close_t = start * prod(1 + r_i for i <= t)`.
- Open: `open_0 = start`, `open_t = close_{t-1}`.
- High/low: `max(open, close) * 1.003` and `min(open, close) * 0.997`.
- Volume: `rng.integers(1_000_000, 50_000_000)`.
- `RATE` additionally clips close/high/low to `[0.5, 6.0]` and sets volume to 0.
- Values are written with six decimal places; rows are written with plain
  Python formatting so re-running the script produces byte-identical files.

| File | Drift | Vol | Start |
| --- | ---: | ---: | ---: |
| `ASSET_A.csv` | 0.0006 | 0.018 | 100 |
| `ASSET_B.csv` | 0.0003 | 0.014 | 90 |
| `ASSET_C.csv` | 0.0002 | 0.016 | 120 |
| `ASSET_D.csv` | 0.0005 | 0.022 | 80 |
| `ASSET_E.csv` | 0.0004 | 0.020 | 150 |
| `INDEX_A.csv` | 0.0003 | 0.009 | 1000 |
| `INDEX_B.csv` | 0.0002 | 0.008 | 4000 |
| `RATE.csv` | 0.0000 | 0.0004 | 2.5 |

Each file has 500 rows from `2024-06-03` to `2026-05-01` and uses the
yfinance-style layout (`Date,Close,High,Low,Open,Volume` plus a duplicate
ticker row).

## Scope

These files exist only to make the pipeline runnable and testable offline.
Do not present them as market history, do not compute investment conclusions
from them, and do not copy them into any application material as evidence of
performance. `demo.py` output is a mechanical exercise on this synthetic
input.
