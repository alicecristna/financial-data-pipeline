"""DataFetcher must not attempt network access when the optional dependency is absent."""
from core import data_fetcher


def test_fetch_yfinance_returns_empty_without_yfinance(monkeypatch):
    monkeypatch.setattr(data_fetcher, "yf", None)
    fetcher = data_fetcher.DataFetcher()
    out = fetcher.fetch_yfinance(["AAPL"], start="2024-01-01", end="2024-01-05")
    assert out.empty
