"""Round-5 integration regressions: processor fill policy and demo labeling.

The processor tests write temporary CSV data so they can also be run against
commit d69e380, where the default forward fill hides the gap.
"""
import pandas as pd
import pytest

import demo
from core.factor_backtest import FactorBacktest
from core.processor import DataProcessor
from core.risk_metrics import RiskMetrics


def _write_gap_csvs(tmp_path) -> DataProcessor:
    (tmp_path / "ASSET_A.csv").write_text(
        "Date,Close\n2024-01-02,100.0\n2024-01-03,101.0\n2024-01-05,103.0\n",
        encoding="utf-8",
    )
    (tmp_path / "ASSET_B.csv").write_text(
        "Date,Close\n"
        "2024-01-02,50.0\n2024-01-03,51.0\n2024-01-04,52.0\n2024-01-05,53.0\n",
        encoding="utf-8",
    )
    return DataProcessor(str(tmp_path))


def test_default_processing_preserves_gap_and_backtest_rejects_it(tmp_path):
    processor = _write_gap_csvs(tmp_path)
    prices = processor.process_prices()

    # The 2024-01-04 price is missing for ASSET_A and must stay missing.
    assert pd.isna(prices.loc["2024-01-04", "ASSET_A"])
    assert prices["ASSET_A"].isna().sum() == 1

    with pytest.raises(ValueError, match="fixed universe"):
        FactorBacktest(prices)


def test_ffill_is_an_explicit_caller_opt_in(tmp_path):
    processor = _write_gap_csvs(tmp_path)

    default_prices = processor.process_prices()
    assert pd.isna(default_prices.loc["2024-01-04", "ASSET_A"])

    filled = processor.process_prices(fill_method="ffill")
    assert not filled["ASSET_A"].isna().any()
    assert (
        filled.loc["2024-01-04", "ASSET_A"]
        == default_prices.loc["2024-01-03", "ASSET_A"]
    )

    # With the gap explicitly filled by the caller, the fixed universe passes.
    FactorBacktest(filled)


def test_unsupported_fill_method_raises(tmp_path):
    processor = _write_gap_csvs(tmp_path)
    with pytest.raises(ValueError, match="fill_method"):
        processor.process_prices(fill_method="backfill")


def test_demo_risk_summary_is_equal_weight_not_last_asset():
    index = pd.date_range("2024-01-01", periods=6)
    returns = pd.DataFrame(
        {
            "ASSET_A": [0.01, 0.02, -0.01, 0.03, 0.00, 0.01],
            "ASSET_B": [0.00, 0.01, 0.01, -0.02, 0.02, 0.00],
            "ASSET_C": [0.02, -0.01, 0.00, 0.01, 0.01, -0.01],
            "ASSET_D": [0.00, 0.00, 0.01, 0.00, -0.01, 0.02],
            "ASSET_E": [0.50, 0.50, 0.50, 0.50, 0.50, 0.50],
        },
        index=index,
    )

    frame, row = demo.equal_weight_risk_summary(returns, demo.STOCKS)

    assert list(frame.columns) == ["EQUAL_WEIGHT"]
    expected = returns[demo.STOCKS].mean(axis=1, skipna=False).rename("EQUAL_WEIGHT")
    pd.testing.assert_series_equal(frame["EQUAL_WEIGHT"], expected)
    assert not frame["EQUAL_WEIGHT"].equals(
        returns["ASSET_E"].rename("EQUAL_WEIGHT")
    )

    # The row is the EQUAL_WEIGHT series' risk metrics, not the last asset's.
    assert row.name == "EQUAL_WEIGHT"
    equal_weight_vol = RiskMetrics(frame).ewma_volatility().iloc[-1, 0]
    asset_e_vol = RiskMetrics(returns[["ASSET_E"]]).ewma_volatility().iloc[-1, 0]
    assert row["EWMA_Vol_Ann"] == pytest.approx(equal_weight_vol, abs=1e-4)
    assert row["EWMA_Vol_Ann"] != pytest.approx(asset_e_vol, abs=1e-4)
