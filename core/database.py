"""
DatabaseManager — SQLite 行情数据存取
"""

"""主流的防止sql注入的参数化方法有两种，一种是用问号占位符号，另一种是使用字典进行命名占位 """
import sqlite3
import pandas as pd
from pathlib import Path

from .utils import setup_logger

logger = setup_logger(__name__)

class DatabaseManager:
    """SQLite 数据库：建表、索引、写入、查询、聚合"""

    def __init__(self, db_path: str = "financial_data.db"):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.executescript("""
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        PRAGMA cache_size = -64000;
        PRAGMA temp_store = MEMORY;
    """)
        self.conn.commit()

        logger.info(f"数据库连接: {self.db_path}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ========== 建表 + 索引 ==========
    def create_tables(self):
        """建表并创建必要的索引"""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS daily_prices (
                ticker TEXT NOT NULL,
                date   TEXT NOT NULL,
                open   REAL,
                high   REAL,
                low    REAL,
                close  REAL,
                volume INTEGER,
                PRIMARY KEY (ticker, date)
            );

            CREATE TABLE IF NOT EXISTS tickers (
                ticker   TEXT PRIMARY KEY,
                name     TEXT,
                sector   TEXT,
                exchange TEXT
            );

            CREATE TABLE IF NOT EXISTS update_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ticker      TEXT,
                rows_added  INTEGER
            );

            -- 只有 date 需要单独索引
            -- ticker 的查询已被复合主键的左前缀覆盖
            CREATE INDEX IF NOT EXISTS idx_date ON daily_prices(date);
        """)
        self.conn.commit()
        logger.info("数据表和索引创建完成")

    # ========== 写入 ==========
    @staticmethod
    def _prepare_price_rows(df: pd.DataFrame):
        """Validate/normalise a price frame and return (columns, rows)."""
        cols = ["ticker", "date", "open", "high", "low", "close", "volume"]
        available = [c for c in cols if c in df.columns]
        if not available:
            raise ValueError(
                "行情数据缺少可用列，至少需要以下之一: " + ", ".join(cols)
            )

        data = df[available].copy()

        if "date" in data.columns:
            data["date"] = pd.to_datetime(data["date"]).dt.strftime("%Y-%m-%d")

        if "ticker" in data.columns and "date" in data.columns:
            data = data.drop_duplicates(subset=["ticker", "date"], keep="last")

        records = [
            tuple(None if pd.isna(value) else value for value in row)
            for row in data.itertuples(index=False, name=None)
        ]
        return available, records

    def upsert_price_rows(self, df: pd.DataFrame) -> int:
        """Execute ``INSERT OR REPLACE`` rows on ``self.conn`` without committing.

        The caller owns the transaction, e.g.::

            with db.conn:
                db.conn.execute("DELETE FROM daily_prices WHERE ticker = ?", (t,))
                db.upsert_price_rows(tidy)

        Any later failure inside that ``with`` block rolls the whole unit of
        work back. Nothing is committed here.
        """
        available, records = self._prepare_price_rows(df)
        placeholders = ", ".join("?" for _ in available)
        sql = (
            f"INSERT OR REPLACE INTO daily_prices ({', '.join(available)}) "
            f"VALUES ({placeholders})"
        )
        self.conn.executemany(sql, records)
        return len(records)

    def insert_prices(self, df: pd.DataFrame):
        """
        批量写入行情数据（单事务 + 主键冲突时覆盖）

        使用 ``INSERT OR REPLACE``，所以对相同 (ticker, date) 重复调用不会
        抛 UNIQUE 约束错误，也不会产生重复行；后写入的同主键记录会覆盖旧值。
        """
        with self.conn:  # 自动 BEGIN + COMMIT，异常时 ROLLBACK
            written = self.upsert_price_rows(df)

        logger.info(f"写入 {written} 条记录")

    # ========== 查询 ==========
    def get_prices(
        self,
        ticker: str = None,
        start: str = None,
        end: str = None,
    ) -> pd.DataFrame:
        """按条件查询行情"""
        clauses = []
        params = []

        if ticker:
            clauses.append("ticker = ?")
            params.append(ticker)
        if start:
            clauses.append("date >= ?")
            params.append(start)
        if end:
            clauses.append("date <= ?")
            params.append(end)

        where = " AND ".join(clauses) if clauses else "1=1"
        return pd.read_sql_query(
            f"SELECT * FROM daily_prices WHERE {where}",
            self.conn,
            params=params,
        )

    # ========== 聚合 ==========
    def get_summary(self) -> pd.DataFrame:
        """
        在数据库端计算每只股票的基本统计

        为什么在 SQL 端算？数据库引擎用 C 写，聚合运算极快；
        拉回 Python 再 groupby 等于把所有数据搬过来再算，多了一道搬运。
        """
        return pd.read_sql_query("""
            SELECT
                ticker,
                COUNT(*)  AS days,
                ROUND(AVG(close), 2) AS avg_close,
                MAX(close) AS max_close,
                MIN(close) AS min_close,
                AVG(volume) AS avg_volume
            FROM daily_prices
            GROUP BY ticker
        """, self.conn)

    # ========== 关闭 ==========
    def close(self):
        self.conn.close()
        logger.info("数据库连接关闭")

 