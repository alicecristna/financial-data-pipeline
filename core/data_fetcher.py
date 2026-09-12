"""
DataFetcher — 金融数据下载
美股/商品/宏观 → yfinance
A 股            → baostock（全免费）

注意：本模块需要可选的网络依赖（yfinance / baostock）。仓库自带的
离线测试与 demo 不会导入或调用它们。
"""
import pandas as pd
import time
from typing import List

from .utils import setup_logger

try:  # 允许在未安装 yfinance 的离线环境中导入本模块
    import yfinance as yf
except ImportError:  # pragma: no cover - depends on optional dependency
    yf = None

logger = setup_logger(__name__)


class DataFetcher:
    """统一的数据获取入口"""

    # =====================================================
    #  yfinance — 美股 / 商品 / 宏观指标
    # =====================================================

    def fetch_yfinance(
        self,
        tickers: List[str],
        start: str = None,
        end: str = None,
        period: str = None,
        max_retries: int = 3,
        request_delay: float = 1.0
    ) -> pd.DataFrame:
        """
        下载美股、大宗商品、宏观指标

        参数:
            tickers: 代码列表 ["AAPL", "GOOGL"]
            start:   "YYYY-MM-DD"（与 period 二选一）
            end:     "YYYY-MM-DD"
            period:  "1mo", "6mo", "1y"（优先于 start/end）
            request_delay: 每次请求前等待秒数，防止 API 限流
        """
        if yf is None:
            logger.error("yfinance 未安装，无法联网下载；离线场景请使用 data/ 下的本地 CSV。")
            return pd.DataFrame()

        for attempt in range(max_retries):
            try:
                logger.info(f"下载 {tickers}（尝试 {attempt + 1}/{max_retries}）")

                time.sleep(request_delay)

                if period:
                    df = yf.download(tickers, period=period, progress=False)
                else:
                    df = yf.download(tickers, start=start, end=end, progress=False)

                if df.empty:
                    logger.warning("返回空数据，请检查代码")
                    return pd.DataFrame()

                logger.info(f"✅ 成功: {df.shape[0]} 行 × {df.shape[1]} 列")
                return df

            # 在 fetch_yfinance 的 for 循环里，替换原来的 except
            except Exception as e:
    # 网络类错误：延时更长，值得重试
                if isinstance(e, (ConnectionError, TimeoutError)):
                    logger.warning(f"网络错误: {e}")
                    wait = 10
                else:
                    logger.error(f"{type(e).__name__}: {e}")
                    wait = 5

                if attempt < max_retries - 1:
                    logger.info(f"等待 {wait} 秒后重试...")
                    time.sleep(wait)
                else:
                    logger.error(f"重试 {max_retries} 次均失败，放弃")
                    return pd.DataFrame()

    def fetch_macro(self, indicators: List[str], period: str = "6mo") -> pd.DataFrame:
        """
        下载宏观/市场指标

        常用代码:
            ^GSPC    标普500
            ^TNX     10年期美债收益率
            ^VIX     VIX 恐慌指数
            GC=F     黄金期货
            CL=F     原油期货
            DX-Y.NYB 美元指数
        """
        return self.fetch_yfinance(indicators, period=period)

    def get_prices(self, df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
        """
        从 yfinance 的 MultiIndex 列中提取价格矩阵

        输入: MultiIndex 列 (Close, AAPL), (Open, GOOGL)
        输出: 行=日期, 列=股票代码 的纯价格 DataFrame
        """
        if df.empty:
            return pd.DataFrame()

        prices = df[price_col].copy()
        prices.columns = [str(c) for c in prices.columns]

        if prices.index.tz is not None:
            prices.index = prices.index.tz_localize(None)

        return prices

    # =====================================================
    #  baostock — A 股日线（全免费，零门槛）
    # =====================================================

    def fetch_a_shares(
        self,
        symbols: List[str],
        start: str = None,
        end: str = None,
        adjust: str = "2",
        request_delay: float = 1.0
    ) -> pd.DataFrame:
        """
        下载 A 股日线数据

        参数:
            symbols: baostock 格式的代码列表
                     深圳 .sz 后缀 → "sz.000001", "sz.000858"
                     上海 .sh 后缀 → "sh.600519", "sh.601398"
            start:   "YYYY-MM-DD" 或 "YYYYMMDD"
            end:     同上
            adjust:  "1"=后复权, "2"=前复权（默认）, "3"=不复权
            request_delay: 每只股票请求前等待秒数，防止 API 限流

        返回:
            行=日期, 列=股票代码 的收盘价 DataFrame
        """
        try:
            import baostock as bs
        except ImportError:
            logger.error("请安装 baostock: pip install baostock")
            return pd.DataFrame()

        # 统一日期格式 → YYYY-MM-DD
        start = (start or "2023-01-01").replace("-", "")[:8]
        end = (end or pd.Timestamp.today().strftime("%Y%m%d")).replace("-", "")[:8]
        # baostock 需要 YYYY-MM-DD
        start = f"{start[:4]}-{start[4:6]}-{start[6:8]}"
        end   = f"{end[:4]}-{end[4:6]}-{end[6:8]}"

        # 登录（一次）
        lg = bs.login()
        if lg.error_code != "0":
            logger.error(f"baostock 登录失败: {lg.error_msg}")
            return pd.DataFrame()

        prices = {}

        for sym in symbols:
            try:
                logger.info(f"baostock 下载 {sym}: {start} → {end}")

                time.sleep(request_delay)
                rs = bs.query_history_k_data_plus(
                    sym,
                    "date,close",
                    start_date=start,
                    end_date=end,
                    frequency="d",
                    adjustflag=adjust
                )

                # baostock 对无效代码也不报错，只返回空的 ResultData
                if rs.error_code != "0":
                    logger.warning(f"{sym}: {rs.error_msg}")
                    continue

                # 遍历结果
                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())

                if not rows:
                    logger.warning(f"{sym}: 无数据")
                    continue

                df = pd.DataFrame(rows, columns=rs.fields)
                df["date"] = pd.to_datetime(df["date"])
                df["close"] = df["close"].astype(float)

                prices[sym] = df.set_index("date")["close"]
                logger.info(f"✅ {sym}: {len(df)} 行")

            except Exception as e:
                logger.error(f"❌ {sym}: {e}")

        bs.logout()

        if not prices:
            return pd.DataFrame()

        return pd.DataFrame(prices).sort_index()


class IncrementalFetcher:
    """
    增量数据更新器

    首次拉取全量历史，之后只拉新增部分，自动合并去重。
    """

    def __init__(self, fetcher: DataFetcher):
        self.fetcher = fetcher
        self.data = {}  # 缓存: {'stocks': DataFrame, 'a_shares': DataFrame, ...}

    def update(
        self,
        asset_type: str,   # 'stocks' | 'macro' | 'a_shares'
        tickers: list,
        start: str = "2020-01-01",
        end: str = None
    ) -> pd.DataFrame:
        """
        增量更新数据

        asset_type: 数据类别
            stocks   → 美股（yfinance）
            macro    → 宏观指标（yfinance）
            a_shares → A 股（baostock）
        """
        end = end or pd.Timestamp.now().strftime("%Y-%m-%d")

        old = self.data.get(asset_type)

        if old is not None and not old.empty:
            last_date = old.index.max()
            new_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

            if new_start >= end:
                logger.info(f"{asset_type} 已是最新，跳过")
                return old

            logger.info(f"{asset_type} 增量: {new_start} → {end}")
        else:
            new_start = start
            logger.info(f"{asset_type} 全量: {start} → {end}")

        # 拉取新数据
        new = self._download(asset_type, tickers, new_start, end)

        if new.empty:
            logger.warning(f"{asset_type} 未获取到新数据")
            return old if old is not None else pd.DataFrame()

        # 合并 + 去重
        if old is not None and not old.empty:
            merged = pd.concat([old, new])
            merged = merged[~merged.index.duplicated(keep="last")]
            merged = merged.sort_index()
            logger.info(f"{asset_type}: {len(old)} + {len(new)} → {len(merged)} 行")
        else:
            merged = new

        self.data[asset_type] = merged
        return merged

    # ---- 内部方法 ----

    def _download(self, asset_type, tickers, start, end):
        """根据类别调用正确的下载方法"""
        if asset_type in ("stocks", "macro"):
            df = self.fetcher.fetch_yfinance(tickers, start=start, end=end)
            return self.fetcher.get_prices(df)

        if asset_type == "a_shares":
            return self.fetcher.fetch_a_shares(tickers, start=start, end=end)

        raise ValueError(f"不支持的资产类型: {asset_type}")