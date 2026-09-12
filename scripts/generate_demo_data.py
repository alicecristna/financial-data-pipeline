"""Generate the deterministic synthetic CSV snapshot used by ``data/``.

The files are NOT real market data and carry no economic meaning. They are
reproducible geometric random walks driven entirely by a fixed seed so that
the offline demo and tests do not depend on any downloaded snapshot.

Usage (from the repository root)::

    python scripts/generate_demo_data.py

The script writes the CSVs and prints a short summary. Re-running it should
produce byte-identical files.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SEED = 20240603
N_DAYS = 500
START_DATE = "2024-06-03"

# Neutral synthetic names with their daily simple-return drift/volatility and
# starting price. "RATE" is a synthetic short-rate series in percent.
SPECS = {
    "ASSET_A": {"drift": 0.0006, "vol": 0.018, "start": 100.0},
    "ASSET_B": {"drift": 0.0003, "vol": 0.014, "start": 90.0},
    "ASSET_C": {"drift": 0.0002, "vol": 0.016, "start": 120.0},
    "ASSET_D": {"drift": 0.0005, "vol": 0.022, "start": 80.0},
    "ASSET_E": {"drift": 0.0004, "vol": 0.020, "start": 150.0},
    "INDEX_A": {"drift": 0.0003, "vol": 0.009, "start": 1000.0},
    "INDEX_B": {"drift": 0.0002, "vol": 0.008, "start": 4000.0},
    "RATE": {"drift": 0.0000, "vol": 0.0004, "start": 2.5},
}


def _rng_for(index: int) -> np.random.Generator:
    # A per-series seed derived from the fixed SEED keeps the files
    # independent but fully reproducible.
    return np.random.default_rng([SEED, index])


def generate_series(ticker: str, spec: dict, index: int) -> pd.DataFrame:
    rng = _rng_for(index)
    simple_returns = rng.normal(spec["drift"], spec["vol"], N_DAYS)
    close = spec["start"] * np.cumprod(1.0 + simple_returns)

    open_ = np.empty(N_DAYS)
    open_[0] = spec["start"]
    open_[1:] = close[:-1]

    high = np.maximum(open_, close) * 1.003
    low = np.minimum(open_, close) * 0.997
    volume = rng.integers(1_000_000, 50_000_000, N_DAYS)

    if ticker == "RATE":
        close = np.clip(close, 0.5, 6.0)
        high = np.clip(high, 0.5, 6.0)
        low = np.clip(low, 0.5, 6.0)
        volume = np.zeros(N_DAYS, dtype=np.int64)

    dates = pd.bdate_range(START_DATE, periods=N_DAYS)
    return pd.DataFrame(
        {
            "Date": [d.strftime("%Y-%m-%d") for d in dates],
            "Close": close,
            "High": high,
            "Low": low,
            "Open": open_,
            "Volume": volume,
        }
    )


def write_csv(ticker: str, frame: pd.DataFrame) -> Path:
    path = DATA_DIR / f"{ticker}.csv"
    lines = [
        "Date,Close,High,Low,Open,Volume",
        "," + ",".join([ticker] * 5),
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"{row.Date},{row.Close:.6f},{row.High:.6f},"
            f"{row.Low:.6f},{row.Open:.6f},{int(row.Volume)}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"seed={SEED} days={N_DAYS} start={START_DATE}")
    for index, (ticker, spec) in enumerate(SPECS.items()):
        frame = generate_series(ticker, spec, index)
        path = write_csv(ticker, frame)
        print(
            f"  {path.name:<12} rows={len(frame)} "
            f"first={frame['Date'].iloc[0]} last={frame['Date'].iloc[-1]}"
        )
    print("Synthetic data written. These files are not market data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
