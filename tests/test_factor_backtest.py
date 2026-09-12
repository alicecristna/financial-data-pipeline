"""Factor backtest regressions: signals, equal weight and IC alignment."""
import numpy as np
import pandas as pd
import pytest

from core.factor_backtest import FactorBacktest


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n)


def test_short_threshold_none_never_short():
    factor = pd.DataFrame(
        {"A": [-2.0], "B": [2.0], "C": [0.0]}, index=_index(1)
    )
    bt = FactorBacktest(pd.DataFrame(
        {"A": [100.0], "B": [100.0], "C": [100.0]}, index=_index(1)
    ))
    signals = bt.generate_signals(factor, long_threshold=0.0, short_threshold=None)
    assert signals.loc[:, "A"].tolist() == [0]
    assert signals.loc[:, "C"].tolist() == [0]
    assert signals.to_numpy().min() >= 0


def test_explicit_short_threshold_creates_short_signal():
    factor = pd.DataFrame({"A": [-2.0], "B": [2.0]}, index=_index(1))
    bt = FactorBacktest(pd.DataFrame({"A": [100.0], "B": [100.0]}, index=_index(1)))
    signals = bt.generate_signals(factor, long_threshold=0.0, short_threshold=-1.0)
    assert signals.loc[:, "A"].tolist() == [-1]
    assert signals.loc[:, "B"].tolist() == [1]


def test_equal_weight_portfolio_uses_cross_sectional_mean():
    prices = pd.DataFrame(
        {"A": [100.0, 101.0, 102.01, 103.0301],
         "B": [100.0, 103.0, 106.09, 109.2727]},
        index=_index(4),
    )
    bt = FactorBacktest(prices)
    bt.signals = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
    bt.calculate_strategy_returns()
    perf = bt.calculate_performance()

    # Simple daily returns are exactly 1% and 3%, so the equal-weight
    # portfolio compounds 2% per day for three valid periods.
    expected_total = (1.0 + 0.02) ** 3 - 1.0
    expected_annual = (1.0 + expected_total) ** (252 / 3) - 1.0
    sum_total = (1.0 + 0.04) ** 3 - 1.0

    assert perf["total_return"] == pytest.approx(expected_total)
    assert perf["annual_return"] == pytest.approx(expected_annual)
    assert perf["total_return"] != pytest.approx(sum_total)


def test_performance_sharpe_is_annualised_periodic_ratio():
    # Fixture reconstructed from the independent audit values so that the
    # legacy formula (annual_return / annual_vol) yields 12.2457767 while the
    # corrected formula (mean / std * sqrt(252)) yields 5.4751891. It is a
    # formula regression fixture, not a market result or a performance claim.
    mu = 1.0 / 175.0
    legacy_target = 12.2457767
    sigma0 = (np.exp(252 * mu) - 1) / (legacy_target * np.sqrt(252))
    pattern = np.concatenate([np.ones(126), -np.ones(126)])
    returns = mu + sigma0 * (pattern - pattern.mean()) / pattern.std(ddof=0)

    bt = FactorBacktest(pd.DataFrame({"A": [100.0, 101.0]}, index=_index(2)))
    bt.strategy_returns = pd.DataFrame({"A": returns}, index=_index(len(returns)))
    perf = bt.calculate_performance()

    legacy = (np.exp(252 * returns.mean()) - 1) / (
        returns.std(ddof=0) * np.sqrt(252)
    )
    corrected = returns.mean() / returns.std(ddof=0) * np.sqrt(252)

    assert legacy == pytest.approx(legacy_target, rel=1e-6)
    assert corrected == pytest.approx(5.4751891, rel=1e-6)
    assert perf["sharpe_ratio"] == pytest.approx(corrected)
    assert perf["sharpe_ratio"] != pytest.approx(legacy)


def _rank_returns_from_factor(factor: pd.DataFrame) -> pd.DataFrame:
    """Build returns whose t+1 cross-section ranks with factor row t."""
    returns = pd.DataFrame(0.0, index=factor.index, columns=factor.columns)
    for t in range(len(factor) - 1):
        ranks = factor.iloc[t].rank()
        returns.iloc[t + 1] = (ranks - ranks.mean()) / 100.0
    return returns


def test_ic_matches_same_period_forward_returns():
    rng = np.random.default_rng(0)
    t_len, n_assets = 8, 12
    idx = _index(t_len)
    factor = pd.DataFrame(
        rng.normal(size=(t_len, n_assets)),
        index=idx,
        columns=[f"A{i}" for i in range(n_assets)],
    )
    returns = _rank_returns_from_factor(factor)
    prices = np.exp(returns.cumsum()) * 100
    bt = FactorBacktest(prices)

    ic = bt.calculate_ic(factor)

    assert len(ic) == t_len - 1
    assert ic.index.max() == idx[-2]
    assert ic.iloc[0] == pytest.approx(1.0)
    assert ic.mean() == pytest.approx(1.0)


def test_ic_detects_double_shift_regression():
    # returns[t+1] ranks exactly with factor[t]; returns[t+2] anti-ranks with
    # factor[t]. If the implementation looks up forward_returns at the *next*
    # date (double shift), the first IC flips to -1.
    rng = np.random.default_rng(1)
    t_len, n_assets = 5, 12
    idx = _index(t_len)
    factor = pd.DataFrame(
        rng.normal(size=(t_len, n_assets)),
        index=idx,
        columns=[f"A{i}" for i in range(n_assets)],
    )
    ranks = factor.iloc[0].rank()
    centered = ranks - ranks.mean()
    returns = pd.DataFrame(0.0, index=idx, columns=factor.columns)
    returns.iloc[1] = centered / 100.0
    returns.iloc[2] = -centered / 100.0
    prices = np.exp(returns.cumsum()) * 100
    bt = FactorBacktest(prices)

    ic = bt.calculate_ic(factor)
    assert ic.loc[idx[0]] == pytest.approx(1.0)


def test_ic_excludes_last_date_without_forward_return():
    rng = np.random.default_rng(2)
    t_len, n_assets = 6, 12
    idx = _index(t_len)
    factor = pd.DataFrame(
        rng.normal(size=(t_len, n_assets)),
        index=idx,
        columns=[f"A{i}" for i in range(n_assets)],
    )
    returns = _rank_returns_from_factor(factor)
    prices = np.exp(returns.cumsum()) * 100
    bt = FactorBacktest(prices)

    ic = bt.calculate_ic(factor)
    assert idx[-1] not in ic.index
