"""Offline demo for the financial data pipeline.

Runs entirely on the deterministic synthetic CSVs under ``data/`` (see
``data/README.md`` and ``scripts/generate_demo_data.py``). It does not import
yfinance/baostock and makes no network calls, so it is safe to run in CI and
without credentials.

Usage:
    python demo.py
"""
from pathlib import Path

from core.factor_backtest import FactorBacktest
from core.processor import DataProcessor, direction_label
from core.risk_metrics import RiskMetrics, RiskReport

ROOT = Path(__file__).resolve().parent
STOCKS = ["ASSET_A", "ASSET_B", "ASSET_C", "ASSET_D", "ASSET_E"]
BENCHMARKS = ["INDEX_A", "INDEX_B"]
RATE_TICKER = "RATE"


def _load_prices():
    processor = DataProcessor(str(ROOT / "data"))
    raw = processor.load_raw()
    if raw.empty:
        raise SystemExit("data/ 目录下没有可读取的 CSV 文件")

    prices = processor.process_prices(fill_method=None)
    # Rate series is not an investable price; keep it out of the equity matrix.
    prices = prices.loc[:, STOCKS + BENCHMARKS]
    return raw, prices


def equal_weight_risk_summary(returns, stocks):
    """Latest risk metrics for one equal-weight stock portfolio.

    Returns ``(frame, row)`` where ``frame`` is the exact one-column
    ``EQUAL_WEIGHT`` return series fed to ``RiskMetrics`` and ``row`` is that
    series' row from ``RiskReport.generate_summary``. Exposing both lets the
    caller and tests verify that the summary is the portfolio's, not a single
    asset's.
    """
    series = returns[stocks].mean(axis=1, skipna=False).rename("EQUAL_WEIGHT")
    frame = series.to_frame()
    summary = RiskReport(RiskMetrics(frame)).generate_summary()
    return frame, summary.loc["EQUAL_WEIGHT"]


def main() -> int:
    raw, prices = _load_prices()
    returns = prices.pct_change(fill_method=None).dropna(how="all").dropna(axis=1, how="all")

    print("=" * 66)
    print("Financial Data Pipeline - offline demo (no network calls)")
    print("=" * 66)
    print(f"CSV files loaded      : {sorted(raw['ticker'].unique())}")
    print(f"Price matrix          : {prices.shape[0]} rows x {prices.shape[1]} cols")
    print(f"Date range            : {prices.index.min().date()} -> {prices.index.max().date()}")
    print()

    print("Risk summary (latest available date, equal-weight portfolio)")
    print("-" * 66)
    equal_weight, risk_row = equal_weight_risk_summary(returns, STOCKS)
    for name, value in risk_row.items():
        print(f"  {name:<20}: {value}")
    print()

    print("20-day momentum factor: next-day Rank IC (all assets)")
    print("-" * 66)
    factor = prices.pct_change(20, fill_method=None)
    backtest = FactorBacktest(prices)
    ic = backtest.calculate_ic(factor, min_obs=3)
    print(f"  IC observations     : {len(ic)}")
    print(f"  Mean IC              : {ic.mean():.4f}" if len(ic) else "  Mean IC              : n/a")
    print()

    print("Equal-weight long-only backtest of the momentum factor")
    print("-" * 66)
    signals = backtest.generate_signals(factor, long_threshold=0.0, short_threshold=None)
    backtest.calculate_strategy_returns(signals)
    perf = backtest.calculate_performance()
    print(f"  Annualized return    : {perf['annual_return']:.2%}")
    print(f"  Annualized volatility: {perf['annual_volatility']:.2%}")
    print(f"  Sharpe (rf=0)        : {perf['sharpe_ratio']:.2f}")
    print(f"  Max drawdown         : {perf['max_drawdown']:.2%}")
    print()

    labels = direction_label(prices[STOCKS[0]].pct_change(fill_method=None))
    print(f"Direction labels for {STOCKS[0]}: {labels.notna().sum()} valid, "
          f"{labels.isna().sum()} missing (final row has no future return)")
    print()
    print("Note: the data is synthetic (fixed-seed random walks, neutral names);")
    print("these numbers are a mechanical computation on it, not investment")
    print("advice, not a performance claim and not a market result.")
    print("DEMO OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
