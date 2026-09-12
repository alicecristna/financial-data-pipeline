"""
DataPipeline — 端到端数据管道
下载 → 格式转换 → 写入数据库 → 记录日志
"""
import pandas as pd
from datetime import datetime, timedelta

from .data_fetcher import DataFetcher
from .database import DatabaseManager
from .utils import setup_logger

logger = setup_logger(__name__)


class DataPipeline:
    """串联 API 拉取与数据库存储"""

    def __init__(self, db_path: str = "financial_data.db", fetcher=None):
        self.fetcher = fetcher if fetcher is not None else DataFetcher()
        self.db = DatabaseManager(db_path)
        self.db.create_tables()

    # ========== 核心：更新行情 ==========
    def update_daily_prices(
        self,
        tickers: list,
        start: str = None,
        end: str = None,
        years_back: int = 2,
    ):
        """
        下载行情 → 写入数据库

        幂等策略：先删该股票的全部旧数据，再写入新数据。
        同一只股票跑十次，结果和跑一次一样。

        原子性：删除、插入与 update_log 写入共用同一个 SQLite 事务。
        任一步失败（例如日志写入被触发器拒绝）会整体回滚，旧数据保持不变。
        这里刻意使用直接 SQL（``executemany``），因为 pandas ``to_sql``
        可能自行提交，从而破坏事务边界。
        """
        if start is None:
            start = (datetime.now() - timedelta(days=365 * years_back)).strftime("%Y-%m-%d")
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"更新 {len(tickers)} 只股票: {start} → {end}")

        for ticker in tickers:
            try:
                # 1. 下载
                raw = self.fetcher.fetch_yfinance([ticker], start=start, end=end)
                prices = self.fetcher.get_prices(raw)

                if prices.empty:
                    logger.warning(f"{ticker}: 无数据，跳过")
                    continue

                # 2. 转为 (ticker, date, close) 三列格式
                tidy = self._to_tidy(prices)
                tidy = tidy.drop_duplicates(subset=["ticker", "date"], keep="last")

                # 3. 写入：先删旧数据，再插入（幂等）
                #    删除 + 插入 + 日志在同一个事务内完成；任何一步失败都会
                #    回滚，不会留下半更新的行情数据。
                with self.db.conn:
                    self.db.conn.execute(
                        "DELETE FROM daily_prices WHERE ticker = ?", (ticker,)
                    )
                    count = self.db.upsert_price_rows(tidy)
                    # 更新日志（append-only 审计日志，重复运行会多出一条日志）
                    self.db.conn.execute(
                        "INSERT INTO update_log (ticker, rows_added) VALUES (?, ?)",
                        (ticker, count),
                    )

                logger.info(f"✅ {ticker}: {count} 行")

            except Exception as e:
                logger.error(f"❌ {ticker} 失败: {e}")
                # 单只失败不影响其他股票，继续下一个

    # ========== 工具：格式转换 ==========
    def _to_tidy(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        价格矩阵 → 三列表

        输入: 行=日期, 列=股票代码, 值=收盘价
        输出: ticker | date | close
        """
        tidy = prices.reset_index()
        tidy = tidy.rename(columns={tidy.columns[0]: "Date"})
        tidy = tidy.melt(
            id_vars="Date",
            var_name="ticker",
            value_name="close",
        )
        tidy["date"] = pd.to_datetime(tidy["Date"]).dt.strftime("%Y-%m-%d")
        return tidy[["ticker", "date", "close"]]

    # ========== 查询 ==========
    def get_price_matrix(self, ticker: str = None) -> pd.DataFrame:
        """
        从数据库读取价格矩阵（行=日期, 列=股票代码）

        和 DataFetcher.get_prices 返回格式一致，可直接交给 FactorBacktest。
        """
        df = self.db.get_prices(ticker=ticker)
        if df.empty:
            return pd.DataFrame()

        return df.pivot(index="date", columns="ticker", values="close").sort_index()

    def close(self):
        self.db.close()