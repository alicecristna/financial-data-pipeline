"""utils.dtype optimisation must not silently alter values."""
import pandas as pd

from core.utils import optimize_dtypes


def test_optimize_dtypes_preserves_float_precision():
    df = pd.DataFrame(
        {
            "close": [123.45678901234567, 98.76543210987654],
            "volume": [12345678, 23456789],
            "ticker": ["ASSET_A", "ASSET_A"],
        }
    )
    out = optimize_dtypes(df)
    assert out["close"].iloc[0] == 123.45678901234567
    assert out["close"].dtype == "float64"
    assert out["volume"].iloc[1] == 23456789
    assert out["ticker"].dtype.name == "category"


def test_optimize_dtypes_preserves_high_cardinality_strings():
    # Under pandas >= 3 the default string dtype is StringDtype, not object;
    # values must be preserved without asserting a specific dtype.
    df = pd.DataFrame({"name": [f"id-{i}" for i in range(10)]})
    out = optimize_dtypes(df)
    assert out["name"].tolist() == df["name"].tolist()
    assert pd.api.types.is_string_dtype(out["name"])
    assert not isinstance(out["name"].dtype, pd.CategoricalDtype)


def test_optimize_dtypes_categoricalises_pandas_string_dtype():
    df = pd.DataFrame(
        {"ticker": pd.array(["ASSET_A"] * 3 + ["ASSET_B"] * 3, dtype="string")}
    )
    out = optimize_dtypes(df)
    assert isinstance(out["ticker"].dtype, pd.CategoricalDtype)
    assert out["ticker"].astype(str).tolist() == ["ASSET_A"] * 3 + ["ASSET_B"] * 3
