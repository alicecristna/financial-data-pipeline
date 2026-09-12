"""CSV header-format handling and direction-label correctness."""
import codecs
from pathlib import Path

import pandas as pd
import pytest

from core.processor import DataProcessor, direction_label

FIXTURES = Path(__file__).parent / "fixtures"
EXPECTED_CLOSE = [185.0, 186.5, 184.9, 188.1, 189.3]


def _load_fixtures() -> pd.DataFrame:
    processor = DataProcessor(str(FIXTURES))
    return processor.load_raw()


def test_loader_supports_with_and_without_ticker_row():
    raw = _load_fixtures()
    assert set(raw["ticker"].unique()) == {
        "with_ticker_row",
        "no_ticker_row",
        "rate_style",
    }

    for ticker in ("with_ticker_row", "no_ticker_row"):
        part = raw.loc[raw["ticker"] == ticker].sort_values("date")
        assert len(part) == 5, f"{ticker}: first or last row was dropped"
        assert part["date"].min() == pd.Timestamp("2024-01-02")
        assert part["Close"].tolist() == pytest.approx(EXPECTED_CLOSE)


def test_loader_supports_rate_price_ticker_preamble():
    raw = _load_fixtures()
    part = raw.loc[raw["ticker"] == "rate_style"].sort_values("date")
    assert len(part) == 5
    assert part["date"].min() == pd.Timestamp("2024-01-02")
    assert part["Close"].tolist() == pytest.approx([5.31, 5.32, 5.30, 5.31, 5.29])


def test_loader_handles_utf8_bom(tmp_path):
    content = "Date,Close\n2024-01-02,1.0\n2024-01-03,2.0\n"
    (tmp_path / "bom.csv").write_bytes(codecs.BOM_UTF8 + content.encode("utf-8"))
    raw = DataProcessor(str(tmp_path)).load_raw()
    assert len(raw) == 2
    assert raw["Close"].tolist() == [1.0, 2.0]


def test_loaded_dates_are_parsed_and_sorted_input_is_not_required():
    raw = _load_fixtures()
    assert raw["date"].notna().all()
    assert pd.api.types.is_datetime64_any_dtype(raw["date"])


def test_direction_label_marks_last_row_missing():
    returns = pd.Series(
        [0.01, -0.02, 0.03, -0.01],
        index=pd.date_range("2024-01-01", periods=4),
    )
    label = direction_label(returns)
    assert label.iloc[:3].tolist() == [0.0, 1.0, 0.0]
    assert pd.isna(label.iloc[-1])


def test_direction_label_drops_correct_rows_when_combined_with_features():
    returns = pd.Series(
        [0.01, -0.02, 0.03, -0.01],
        index=pd.date_range("2024-01-01", periods=4),
    )
    frame = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]}, index=returns.index)
    frame["target"] = direction_label(returns)
    cleaned = frame.dropna()
    assert len(cleaned) == 3
    assert cleaned.index.tolist() == list(returns.index[:3])
