"""Round-3 financial-correctness regressions.

These tests are intentionally self-contained (only ``FactorBacktest`` and
``RiskMetrics``) so the same file can be run against the earlier commit
d9d41eb, where they are expected to fail, and against the fixed tree, where
they pass.
"""
import numpy as np
import pandas as pd
import pytest

from core.factor_backtest import FactorBacktest
from core.risk_metrics import RiskMetrics


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n)


def _long_only(prices: pd.DataFrame) -> FactorBacktest:
    bt = FactorBacktest(prices)
    bt.signals = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
    bt.calculate_strategy_returns()
    return bt


# ---------------------------------------------------------------------------
# 1. Simple-return equal-weight portfolio accounting
# ---------------------------------------------------------------------------

def test_equal_weight_50_50_gain_and_loss_compounds_to_25_percent():
    # One asset +100%, one -50%; a 50/50 basket must gain 25%:
    # 0.5 * (1.00) + 0.5 * (-0.50) = +0.25
    prices = pd.DataFrame(
        {"A": [100.0, 200.0], "B": [100.0, 50.0]}, index=_index(2)
    )
    perf = _long_only(prices).calculate_performance()
    assert perf["total_return"] == pytest.approx(0.25)
    assert perf["cum_returns"].iloc[-1] == pytest.approx(1.25)


def test_equal_weight_uses_simple_returns_not_log_return_means():
    # The same scenario has a mean log return of 0, which would wrongly
    # produce flat wealth if log returns were averaged.
    prices = pd.DataFrame(
        {"A": [100.0, 200.0], "B": [100.0, 50.0]}, index=_index(2)
    )
    perf = _long_only(prices).calculate_performance()
    log_return_mean = (np.log(2.0) + np.log(0.5)) / 2.0
    assert log_return_mean == pytest.approx(0.0)
    assert perf["total_return"] == pytest.approx(0.25)
    assert perf["total_return"] != pytest.approx(np.exp(log_return_mean) - 1.0)


# ---------------------------------------------------------------------------
# 2. Execution and commission timing
# ---------------------------------------------------------------------------

def test_commission_charges_opening_trade_from_zero_position():
    prices = pd.DataFrame({"A": [100.0, 110.0, 110.0]}, index=_index(3))
    bt = FactorBacktest(prices)
    signals = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=_index(3))

    commission = bt.calculate_commission(signals, rate=0.01)
    assert commission["A"].iloc[1] == pytest.approx(0.01)  # opening trade
    assert commission["A"].iloc[2] == pytest.approx(0.0)   # no further trade

    net = bt.calculate_net_returns(signals, commission_rate=0.01)
    assert net["A"].iloc[1] == pytest.approx(0.10 - 0.01)
    assert net["A"].iloc[2] == pytest.approx(0.0)


def test_commission_does_not_charge_final_unexecuted_signal():
    prices = pd.DataFrame({"A": [100.0, 100.0, 100.0]}, index=_index(3))
    bt = FactorBacktest(prices)
    signals = pd.DataFrame({"A": [1.0, 1.0, 0.0]}, index=_index(3))

    commission = bt.calculate_commission(signals, rate=0.01)
    # The final signal would only execute after the sample ends.
    assert commission["A"].iloc[2] == pytest.approx(0.0)
    # The row-0 signal is executed at row 1 and charged there.
    assert commission["A"].iloc[1] == pytest.approx(0.01)


def test_strategy_returns_use_positions_from_previous_signal():
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0]}, index=_index(3))
    bt = FactorBacktest(prices)
    signals = pd.DataFrame({"A": [1.0, 0.0, 0.0]}, index=_index(3))
    bt.calculate_strategy_returns(signals)

    assert bt.strategy_returns["A"].iloc[1] == pytest.approx(0.10)
    assert bt.strategy_returns["A"].iloc[2] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 3. Drawdowns include initial wealth
# ---------------------------------------------------------------------------

def test_factor_backtest_first_return_drawdown_is_not_zero():
    prices = pd.DataFrame({"A": [100.0, 80.0, 80.0]}, index=_index(3))
    perf = _long_only(prices).calculate_performance()
    assert perf["max_drawdown"] == pytest.approx(-0.20)


def test_risk_metrics_current_drawdown_includes_initial_wealth():
    returns = pd.DataFrame({"A": [-0.20, 0.0, 0.05]}, index=_index(3))
    drawdown = RiskMetrics(returns).current_drawdown()["A"]

    assert drawdown.iloc[0] == pytest.approx(-0.20)
    assert drawdown.min() == pytest.approx(-0.20)


def test_risk_metrics_rolling_max_drawdown_includes_initial_wealth():
    returns = pd.DataFrame({"A": [-0.20, 0.0, 0.0, 0.10]}, index=_index(4))
    drawdown = RiskMetrics(returns).rolling_max_drawdown(window=2)

    assert drawdown["A"].iloc[1] == pytest.approx(-0.20)


# ---------------------------------------------------------------------------
# 4. Zero volatility Sharpe is NaN, never infinity
# ---------------------------------------------------------------------------

@pytest.fixture()
def constant_returns() -> pd.DataFrame:
    return pd.DataFrame({"A": [0.01] * 6}, index=_index(6))


def test_rolling_sharpe_zero_volatility_is_nan(constant_returns):
    got = RiskMetrics(constant_returns).rolling_sharpe(window=3, annualize=False)
    assert got.isna().all().all()
    assert not np.isinf(got.to_numpy()).any()


def test_expanding_sharpe_zero_volatility_is_nan(constant_returns):
    got = RiskMetrics(constant_returns).expanding_sharpe()
    assert got.isna().all().all()
    assert not np.isinf(got.to_numpy()).any()


def test_ewma_sharpe_zero_volatility_is_nan(constant_returns):
    got = RiskMetrics(constant_returns).ewma_sharpe(
        lambda_param=0.94, annualize=False
    )
    assert got.isna().all().all()
    assert not np.isinf(got.to_numpy()).any()
