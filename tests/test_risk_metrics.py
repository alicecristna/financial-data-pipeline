"""Risk metric regressions: unbound annual_vol and quantile typo."""
import numpy as np
import pandas as pd
import pytest

from core.risk_metrics import RiskMetrics


@pytest.fixture()
def returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": [0.01, -0.02, 0.03, -0.01, 0.02, -0.03],
            "B": [0.00, 0.01, -0.01, 0.02, -0.01, 0.005],
        },
        index=pd.date_range("2024-01-01", periods=6),
    )


def test_rolling_volatility_non_annualized_returns_values(returns):
    rm = RiskMetrics(returns)
    got = rm.rolling_volatility(window=2, annualize=False)
    expected = returns.rolling(2, min_periods=1).std(ddof=0)
    pd.testing.assert_frame_equal(got, expected)


def test_rolling_volatility_annualized_scales_by_sqrt_252(returns):
    rm = RiskMetrics(returns)
    raw = rm.rolling_volatility(window=2, annualize=False)
    ann = rm.rolling_volatility(window=2, annualize=True)
    pd.testing.assert_frame_equal(ann, raw * np.sqrt(252))


def test_historical_var_without_window_matches_column_quantile(returns):
    rm = RiskMetrics(returns)
    got = rm.historical_var(confidence=0.95)
    expected = returns.quantile(0.05)
    assert got.values == pytest.approx(expected.values)
    assert list(got.index) == list(expected.index)


def test_historical_var_with_window_returns_rolling_quantile(returns):
    rm = RiskMetrics(returns)
    got = rm.historical_var(confidence=0.9, window=2)
    expected = returns.rolling(2).quantile(0.1)
    pd.testing.assert_frame_equal(got, expected)


def test_rolling_sharpe_annualize_false_returns_periodic_ratio(returns):
    rm = RiskMetrics(returns)
    periodic = rm.rolling_sharpe(window=3, annualize=False)
    annual = rm.rolling_sharpe(window=3, annualize=True)

    # Before the fix the annualize flag was ignored and both outputs matched.
    pd.testing.assert_frame_equal(annual, periodic * np.sqrt(252))
    assert not np.allclose(
        periodic.dropna().to_numpy(), annual.dropna().to_numpy()
    )


def test_rolling_sharpe_matches_periodic_formula_with_rf(returns):
    rm = RiskMetrics(returns)
    rf_rate = 0.02
    excess = returns - rf_rate / 252
    mean = excess.rolling(3, min_periods=1).mean()
    std = excess.rolling(3, min_periods=1).std(ddof=0)
    expected = (mean / std).mask(std == 0)  # zero volatility -> NaN
    pd.testing.assert_frame_equal(
        rm.rolling_sharpe(window=3, annualize=False, rf_rate=rf_rate),
        expected,
    )


def test_rolling_beta_is_two_for_double_exposure():
    index = pd.date_range("2024-01-01", periods=8)
    market = pd.Series(
        [0.01, -0.02, 0.03, 0.01, -0.01, 0.02, 0.015, -0.005], index=index
    )
    frame = pd.DataFrame({"MKT": market, "DOUBLE": 2.0 * market})

    beta = RiskMetrics(frame).rolling_beta("DOUBLE", "MKT", window=3).dropna()

    # The old cov(ddof=1) / var(ddof=0) mix returned 3.0 for this input.
    assert len(beta) == 6
    assert beta.to_numpy() == pytest.approx(np.full(len(beta), 2.0))


def test_ewma_sharpe_uses_centred_variance_not_rms():
    returns = pd.DataFrame(
        {"A": [0.01, 0.03, 0.02]},
        index=pd.date_range("2024-01-01", periods=3),
    )
    rm = RiskMetrics(returns)

    # lambda=0.5 -> alpha=0.5, so with adjust=False:
    #   E[x]   = 0.01, 0.02, 0.02
    #   E[x^2] = 0.0001, 0.0005, 0.00045
    #   Var    = 0, 0.0001, 0.00005
    got = rm.ewma_sharpe(lambda_param=0.5, annualize=False)["A"]
    assert got.iloc[1] == pytest.approx(0.02 / np.sqrt(0.0001))
    assert got.iloc[2] == pytest.approx(0.02 / np.sqrt(0.00005))

    # The RMS variant would have produced these values instead.
    rms_1 = 0.02 / np.sqrt(0.0005)
    assert not np.isclose(got.iloc[1], rms_1)

    annual = rm.ewma_sharpe(lambda_param=0.5, annualize=True)["A"]
    assert annual.iloc[1] == pytest.approx(got.iloc[1] * np.sqrt(252))
