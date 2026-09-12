"""utils.py — 日志配置 & dtype 优化"""
import logging
from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)

def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    """日志配置"""
    if log_file is None:
        log_file = str(BASE_DIR / "logs" / "pipeline.log")

    
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(console)
    logger.addHandler(fh)
    return logger

def _is_text_dtype(dtype) -> bool:
    """True for object-backed strings and for pandas ``StringDtype``.

    pandas 3 infers string columns as ``StringDtype`` by default, while
    older versions use ``object``; both must be treated as text.
    """
    return isinstance(dtype, pd.StringDtype) or dtype == object


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast integer columns and categoricalise low-cardinality text columns.

    Float columns are intentionally left at their original precision: many of
    them hold prices, and silently rounding them to float32 would alter data.
    Text detection supports both ``object`` dtype (pandas < 3) and
    ``StringDtype`` (pandas >= 3) so the same behaviour holds on both.
    """
    out = df.copy()
    for col in out.columns:
        series = out[col]
        if pd.api.types.is_integer_dtype(series):
            out[col] = pd.to_numeric(series, downcast="integer")
        elif _is_text_dtype(series.dtype):
            if series.nunique(dropna=False) < max(2, len(series) // 2):
                out[col] = series.astype("category")
    return out


def validate_price_data(df: pd.DataFrame, name: str = ""):
    """返回 (是否通过, 错误信息)"""
    label = f"[{name}] " if name else ""

    if df.empty:
        return False, f"{label}数据为空"
    if df.isna().all(axis=1).any():
        bad_rows = df[df.isna().all(axis=1)].index[:3].tolist()
        return False, f"{label}全 NaN 行: {bad_rows}"
    if not df.index.is_monotonic_increasing:
        return False, f"{label}日期不单调"
    now = pd.Timestamp.now()
    if df.index.tz is not None:
        now = now.tz_localize(df.index.tz)
    if df.index.max() > now:
        return False, f"{label}存在未来日期: {df.index.max()}"

    return True, ""