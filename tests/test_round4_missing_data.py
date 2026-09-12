"""Round-4 regressions: missing-data policy, per-window drawdown, NaN signals.

Self-contained (``FactorBacktest``/``RiskMetrics`` only) so the same file can
also be run against commit dfe05a0, where the regressions are expected to fail.
"""
import numpy as np
import pandas as pd
import pytest

from core.factor_backtest import FactorBacktest
from core.risk_metrics import RiskMetrics


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n)


# ---------------------------------------------------------------------------
# 1. Fixed-universe missing-data policy (identical on pandas 2.x and 3.x)
# ---------------------------------------------------------------------------

def test_missing_middle_price_raises_instead_of_forward_filling():
    # pandas 2 defaults to fill_method='pad'; pandas 3 does not. Either way,
    # an explicit fill_method=None plus validation must reject the input.
    prices = pd.DataFrame(
        {
            "A": [100.0, 101.0, np.nan, 103.0, 104.0],
            "B": [100.0, 102.0, 104.0, 106.0, 108.0],
        },
        index=_index(5),
    )
    with pytest.raises(ValueError, match="fixed universe"):
        FactorBacktest(prices)


def test_first_return_row_may_be_missing():
    prices = pd.DataFrame(
        {"A": [100.0, 101.0, 102.0], "B": [50.0, 51.0, 52.0]},
        index=_index(3),
    )
    bt = FactorBacktest(prices)

    assert bt.returns.iloc[0].isna().all()
    assert np.isfinite(bt.returns.iloc[1:].to_numpy()).all()


def test_performance_rejects_interior_missing_strategy_returns():
    prices = pd.DataFrame({"A": [100.0, 101.0, 102.0, 103.0]}, index=_index(4))
    bt = FactorBacktest(prices)
    bt.signals = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
    bt.calculate_strategy_returns()

    bt.strategy_returns.iloc[2, 0] = np.nan  # would be silently skipped before
    with pytest.raises(ValueError, match="fixed universe"):
        bt.calculate_performance()


# ---------------------------------------------------------------------------
# 2. Per-window rolling drawdown
# ---------------------------------------------------------------------------

def test_rolling_max_drawdown_resets_each_window():
    # The final window contains only 0% returns, so its drawdown must be 0%,
    # not the -20% carried from the earlier window.
    returns = pd.DataFrame({"A": [-0.20, 0.0, 0.0, 0.0]}, index=_index(4))
    drawdown = RiskMetrics(returns).rolling_max_drawdown(window=2)["A"]
    assert drawdown.iloc[-1] == pytest.approx(0.0, abs=1e-12)


def test_rolling_max_drawdown_compounds_returns_inside_one_window():
    returns = pd.DataFrame({"A": [-0.10, -0.10]}, index=_index(2))
    drawdown = RiskMetrics(returns).rolling_max_drawdown(window=2)["A"]
    assert drawdown.iloc[-1] == pytest.approx(-0.19)


# ---------------------------------------------------------------------------
# 3. Missing signals are flat, with one shared executed-position source
# ---------------------------------------------------------------------------

def _nan_signal_case():
    prices = pd.DataFrame(
        {"A": [100.0, 110.0, 121.0, 133.1]}, index=_index(4)
    )
    signals = pd.DataFrame({"A": [1.0, np.nan, 1.0, 1.0]}, index=_index(4))
    return FactorBacktest(prices), signals


def test_missing_signal_is_treated_as_flat():
    bt, signals = _nan_signal_case()

    positions = bt._executed_positions(signals)
    assert positions["A"].tolist() == [0.0, 1.0, 0.0, 1.0]

    strategy = bt.calculate_strategy_returns(signals)
    pd.testing.assert_frame_equal(strategy, positions * bt.returns)
    assert strategy["A"].iloc[1] == pytest.approx(0.10)
    assert strategy["A"].iloc[2] == pytest.approx(0.0)
    assert strategy["A"].iloc[3] == pytest.approx(0.10)


def test_missing_signal_commission_matches_executed_positions():
    bt, signals = _nan_signal_case()

    commission = bt.calculate_commission(signals, rate=0.01)
    expected = bt._executed_trades(signals).abs() * 0.01
    pd.testing.assert_frame_equal(commission, expected)

    # exit at row 2 and re-entry at row 3 are both charged
    assert commission["A"].tolist() == pytest.approx([0.0, 0.01, 0.01, 0.01])
