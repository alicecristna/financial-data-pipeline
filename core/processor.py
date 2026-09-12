"""
processor.py — 金融数据读取 & 预处理
"""
import os

import numpy as np
import pandas as pd

from .utils import setup_logger, optimize_dtypes

logger = setup_logger(__name__)


def direction_label(returns: pd.Series) -> pd.Series:
    """次日涨跌方向标签。

    标签 t = 1 if returns[t+1] > 0 else 0，returns[t+1] 缺失时为 NaN。
    历史实现直接对 ``returns.shift(-1) > 0`` 取 astype(int)，会把最后一行
    缺失的未来收益错误地标成 0（下跌），从而污染训练集。
    """
    future = returns.shift(-1)
    label = (future > 0).astype("float64")
    label[future.isna()] = np.nan
    return label


class DataProcessor:
    def __init__(self, data_dir="./data/"):
        self.data_dir = data_dir
        self.raw_data = None
        self.prices = None

    # ---------- 任务 10：关键位置日志埋点 ----------
    def load_raw(self, optimize: bool = True):
        """读取 data_dir 下所有 CSV 文件，合并为一个大 DataFrame"""
        df_list = []

        for file in sorted(os.listdir(self.data_dir)):
            if not file.endswith(".csv"):
                continue

            file_path = os.path.join(self.data_dir, file)
            logger.debug("正在读取文件: %s", file_path)

            df = self._read_price_csv(file_path)
            if df.empty:
                logger.warning("文件 %s 没有可用的数据行，跳过", file)
                continue

            ticker = file[: -len(".csv")]
            df["ticker"] = ticker
            df["date"] = df["Date"]
            df_list.append(df)

        if not df_list:
            logger.warning("data_dir 下未找到任何 CSV 文件: %s", self.data_dir)
            return pd.DataFrame()

        self.raw_data = pd.concat(df_list, ignore_index=True)
        logger.info("所有文件读取完成，共 %d 个文件，%d 条记录",
                     len(df_list), len(self.raw_data))

        # 自动优化 dtype
        if optimize:
            self.raw_data = optimize_dtypes(self.raw_data)
            logger.debug("dtype 优化完成")

        return self.raw_data

    @staticmethod
    def _read_price_csv(file_path: str) -> pd.DataFrame:
        """兼容不同来源的行情 CSV（yfinance / baostock 导出）。

        支持的格式：
        1. 标准 yfinance 单票导出：第一行 Date/Close/...，第二行是重复的
           ticker 行（``,AAPL,AAPL,...``），数据从第三行开始；
        2. 普通带表头 CSV：第一行就是表头，没有 ticker 行；
        3. ``^IRX`` 风格的 yfinance 导出：前两行为 ``Price,...`` 和
           ``Ticker,...``，第三行 ``Date,,,,,``，数据从第四行开始。

        实现方式：先按无表头读入，定位真正的表头行，再按位置设置列名，
        最后强制把第一列解析为日期并丢弃所有解析失败的行
        （即 ticker 行/占位行）。
        """
        raw = None
        last_error = None
        for encoding in ("utf-8-sig", "gbk", "gb2312", "latin-1"):
            try:
                raw = pd.read_csv(file_path, header=None, encoding=encoding)
                break
            except UnicodeDecodeError as exc:
                last_error = exc
                logger.warning("编码 %s 失败，尝试下一种编码: %s", encoding, file_path)
        if raw is None:
            raise last_error

        if raw.empty:
            return pd.DataFrame()

        first_cell = str(raw.iat[0, 0]).strip().lower()
        if first_cell in ("price", "ticker"):
            # ^IRX 风格：向下找到 "Date" 行作为数据起始，列名取第一行
            header_idx = 0
            for i in range(min(3, len(raw))):
                if str(raw.iat[i, 0]).strip().lower() == "date":
                    header_idx = i
                    break
            column_names = raw.iloc[0].tolist()
            data = raw.iloc[header_idx + 1:].copy()
        else:
            column_names = raw.iloc[0].tolist()
            data = raw.iloc[1:].copy()

        data.columns = column_names
        data = data.rename(columns={data.columns[0]: "Date"})
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        data = data.dropna(subset=["Date"]).reset_index(drop=True)

        for col in data.columns:
            if col != "Date":
                data[col] = pd.to_numeric(data[col], errors="coerce")

        return data

    def process_prices(self, price_col="Close", fill_method=None):
        """从 raw_data 中提取价格矩阵。

        默认不填充（``fill_method=None``）：缺失价格保留为 NaN 交回调用方，
        由调用方明确决定如何处理。``fill_method="ffill"`` 是显式的调用方
        选择，会用前值填补缺失，从而改变数据语义（例如固定资产池校验
        将不再因缺口报错）。其他取值一律报错。
        """
        if fill_method not in (None, "ffill"):
            raise ValueError(
                f"不支持的 fill_method: {fill_method!r}；"
                "请使用 None（默认，保留缺失）或 'ffill'（显式前向填充）"
            )

        if self.raw_data is None:
            self.load_raw()

        if self.raw_data.empty:
            self.prices = pd.DataFrame()
            return self.prices

        prices = {}
        for ticker, group in self.raw_data.groupby("ticker", observed=True):
            prices[ticker] = group.set_index("date")[price_col]

        self.prices = pd.DataFrame(prices).sort_index()

        if fill_method == "ffill":
            self.prices = self.prices.ffill()

        logger.debug("价格矩阵构建完成，shape=%s", self.prices.shape)
        return self.prices

    def process_returns(self):
        """输出日收益率矩阵"""
        if self.prices is None:
            self.process_prices()
        return self.prices.pct_change(fill_method=None).dropna()

    def get_stats(self):
        """输出基本统计量"""
        if self.prices is None:
            self.process_prices()
        returns = self.process_returns()

        stats = pd.DataFrame({
            "mean": returns.mean(),
            "var": returns.var(),
            "skew": returns.skew(),
        })
        return stats
