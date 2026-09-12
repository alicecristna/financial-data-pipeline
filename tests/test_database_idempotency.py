"""Database write idempotency and offline DataPipeline behaviour."""
import pandas as pd
import pytest

from core.database import DatabaseManager
from core.pipeline import DataPipeline


class FakeFetcher:
    """Stands in for yfinance/baostock; returns pre-built local prices."""

    def __init__(self, prices: pd.DataFrame):
        self._prices = prices

    def fetch_yfinance(self, tickers, start=None, end=None, period=None,
                       max_retries=3, request_delay=1.0):
        return self._prices

    def get_prices(self, raw, price_col="Close"):
        return raw


def _sample_prices() -> pd.DataFrame:
    return pd.DataFrame(
        {"ASSET_A": [100.0, 101.0, 102.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )


def test_insert_prices_twice_is_idempotent(tmp_path):
    db = DatabaseManager(str(tmp_path / "prices.db"))
    db.create_tables()
    df = pd.DataFrame(
        {
            "ticker": ["ASSET_A", "ASSET_A"],
            "date": ["2024-01-02", "2024-01-03"],
            "close": [100.0, 101.0],
        }
    )

    db.insert_prices(df)
    db.insert_prices(df)  # used to raise UNIQUE constraint failed

    stored = db.get_prices("ASSET_A").sort_values("date")
    assert len(stored) == 2
    assert stored["close"].tolist() == [100.0, 101.0]
    db.close()


def test_insert_prices_replaces_same_primary_key(tmp_path):
    db = DatabaseManager(str(tmp_path / "prices.db"))
    db.create_tables()
    first = pd.DataFrame({"ticker": ["ASSET_A"], "date": ["2024-01-02"], "close": [100.0]})
    second = pd.DataFrame({"ticker": ["ASSET_A"], "date": ["2024-01-02"], "close": [123.45]})

    db.insert_prices(first)
    db.insert_prices(second)

    stored = db.get_prices("ASSET_A")
    assert len(stored) == 1
    assert stored["close"].iloc[0] == pytest.approx(123.45)
    db.close()


def test_insert_prices_dates_are_normalised_to_iso(tmp_path):
    db = DatabaseManager(str(tmp_path / "prices.db"))
    db.create_tables()
    db.insert_prices(
        pd.DataFrame(
            {"ticker": ["ASSET_A"], "date": [pd.Timestamp("2024-01-02 15:30:00")], "close": [1.0]}
        )
    )
    assert db.get_prices("ASSET_A")["date"].tolist() == ["2024-01-02"]
    db.close()


def test_pipeline_update_is_idempotent_and_does_not_need_network(tmp_path):
    db_path = str(tmp_path / "pipeline.db")
    pipeline = DataPipeline(db_path, fetcher=FakeFetcher(_sample_prices()))

    pipeline.update_daily_prices(["ASSET_A"], start="2024-01-01", end="2024-01-10")
    first = pipeline.db.get_prices("ASSET_A").sort_values("date")
    pipeline.update_daily_prices(["ASSET_A"], start="2024-01-01", end="2024-01-10")
    second = pipeline.db.get_prices("ASSET_A").sort_values("date")

    assert len(first) == len(second) == 3
    assert first["date"].tolist() == ["2024-01-02", "2024-01-03", "2024-01-04"]
    pd.testing.assert_frame_equal(first, second)

    # update_log is an append-only audit log: one row per update run.
    log = pd.read_sql_query("SELECT COUNT(*) AS n FROM update_log", pipeline.db.conn)
    assert log["n"].iloc[0] == 2
    pipeline.close()


def test_pipeline_keeps_only_latest_duplicate_rows(tmp_path):
    prices = pd.DataFrame(
        {"ASSET_A": [100.0, 100.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-02"]),
    )
    pipeline = DataPipeline(str(tmp_path / "dup.db"), fetcher=FakeFetcher(prices))
    pipeline.update_daily_prices(["ASSET_A"], start="2024-01-01", end="2024-01-10")
    stored = pipeline.db.get_prices("ASSET_A")
    assert len(stored) == 1
    pipeline.close()


def test_pipeline_update_rolls_back_when_update_log_write_fails(tmp_path):
    old = pd.DataFrame(
        {"ticker": ["ASSET_A"], "date": ["2024-01-02"], "close": [100.0]}
    )
    new = pd.DataFrame(
        {"ASSET_A": [999.0, 998.0]},
        index=pd.to_datetime(["2024-01-03", "2024-01-04"]),
    )
    pipeline = DataPipeline(str(tmp_path / "rollback.db"), fetcher=FakeFetcher(new))
    pipeline.db.insert_prices(old)

    # Force the last statement of the delete + insert + log transaction to fail.
    pipeline.db.conn.execute(
        "CREATE TRIGGER fail_update_log BEFORE INSERT ON update_log "
        "BEGIN SELECT RAISE(ABORT, 'forced update_log failure'); END"
    )
    pipeline.db.conn.commit()

    pipeline.update_daily_prices(["ASSET_A"], start="2024-01-01", end="2024-01-10")

    stored = pipeline.db.get_prices("ASSET_A").sort_values("date")
    assert stored["date"].tolist() == ["2024-01-02"]
    assert stored["close"].tolist() == [100.0]

    log = pd.read_sql_query("SELECT COUNT(*) AS n FROM update_log", pipeline.db.conn)
    assert log["n"].iloc[0] == 0
    pipeline.close()
