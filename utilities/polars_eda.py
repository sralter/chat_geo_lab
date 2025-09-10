# polars_eda.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional, Union
import polars as pl
import re

Pathish = Union[str, Path]
Frameish = Union[pl.DataFrame, pl.LazyFrame]

# ---------- helpers ----------


def as_lazy(x: Union[Frameish, Pathish, Iterable[Pathish]]) -> pl.LazyFrame:
    """Accept a LazyFrame/DataFrame, a path, or a list of paths; return a LazyFrame."""
    if isinstance(x, pl.LazyFrame):
        return x
    if isinstance(x, pl.DataFrame):
        return x.lazy()
    if isinstance(x, (str, Path)):
        x = str(x)
        return pl.scan_parquet(x) if x.endswith(".parquet") else pl.scan_csv(x)
    # iterable of paths
    paths = [str(p) for p in x]
    if not paths:
        raise ValueError("No input paths.")
    if all(p.endswith(".parquet") for p in paths):
        return pl.scan_parquet(paths)
    return pl.scan_csv(paths)


def _n_rows(lf: pl.LazyFrame) -> int:
    # Robust across Polars versions: grab the first (and only) value of the single-agg result
    out = lf.select(pl.len()).collect(streaming=True)
    return int(out.to_series(0).item())


# ---------- quick peeks ----------


def peek(lf: Frameish, n: int = 5) -> pl.DataFrame:
    """First n rows (pushes limit down to scan)."""
    lf = as_lazy(lf)
    return lf.limit(n).collect()


def schema(lf: Frameish) -> dict:
    """Column -> dtype dict without collecting data."""
    return as_lazy(lf).schema


def shape(lf: Frameish) -> tuple[int, int]:
    """(rows, cols) — rows computed lazily/streaming."""
    lf = as_lazy(lf)
    return _n_rows(lf), len(lf.columns)


# ---------- missingness & cardinality ----------


def missingness(lf: Frameish, top: Optional[int] = 20) -> pl.DataFrame:
    """Null count & percentage per column, sorted desc by nulls."""
    lf = as_lazy(lf)
    cols = lf.columns
    exprs = [pl.len().alias("_rows")] + [pl.col(c).null_count().alias(c) for c in cols]
    counts = lf.select(exprs).collect(streaming=True)
    n = int(counts["_rows"][0])
    data = (
        pl.DataFrame({"column": cols, "nulls": [int(counts[c][0]) for c in cols]})
        .with_columns(
            pl.col("nulls"), (pl.col("nulls") / pl.lit(n) * 100).alias("null_pct")
        )
        .sort("nulls", descending=True)
    )
    return data.head(top) if top else data


def cardinality(lf: Frameish, top: Optional[int] = 50) -> pl.DataFrame:
    """n_unique per column, sorted desc."""
    lf = as_lazy(lf)
    cols = lf.columns
    exprs = [pl.col(c).n_unique().alias(c) for c in cols]
    out = lf.select(exprs).collect(streaming=True)
    df = pl.DataFrame({"column": cols, "n_unique": [int(out[c][0]) for c in cols]})
    df = df.sort("n_unique", descending=True)
    return df.head(top) if top else df


# ---------- value counts / duplicates ----------


def value_counts(lf: Frameish, column: str, k: int = 20) -> pl.DataFrame:
    """Top-k value counts for a single column (streaming)."""
    lf = as_lazy(lf)
    return (
        lf.group_by(pl.col(column))
        .len()
        .sort(pl.col("len"), descending=True)
        .limit(k)
        .collect(streaming=True)
        .rename({"len": "count"})
    )


def duplicate_rows(
    lf: Frameish, subset: Optional[List[str]] = None, k: int = 20
) -> pl.DataFrame:
    """
    Show most common duplicate groups. Warning: can be heavy with many columns.
    Prefer `subset` (list of columns) for large data.
    """
    lf = as_lazy(lf)
    keys = subset or lf.columns
    return (
        lf.group_by([pl.col(c) for c in keys])
        .len()
        .filter(pl.col("len") > 1)
        .sort(pl.col("len"), descending=True)
        .limit(k)
        .collect(streaming=True)
        .rename({"len": "dupe_count"})
    )


# ---------- numeric summaries ----------


def numeric_summary(lf: Frameish) -> pl.DataFrame:
    """
    Per-numeric-column stats in one pass (mean, std, min, q25, median, q75, max).
    Returns long-form table with one row per column.
    """
    lf = as_lazy(lf)
    num_cols = [c for c, dt in lf.schema.items() if dt in pl.NUMERIC_DTYPES]
    if not num_cols:
        return pl.DataFrame({"column": [], "metric": [], "value": []})
    exprs = []
    for c in num_cols:
        s = pl.col(c)
        exprs += [
            s.mean().alias(f"{c}__mean"),
            s.std(ddof=1).alias(f"{c}__std"),
            s.min().alias(f"{c}__min"),
            s.quantile(0.25).alias(f"{c}__q25"),
            s.median().alias(f"{c}__q50"),
            s.quantile(0.75).alias(f"{c}__q75"),
            s.max().alias(f"{c}__max"),
        ]
    wide = lf.select(exprs).collect(streaming=True)
    long = (
        wide.melt(variable_name="key", value_name="value")
        .with_columns(pl.col("key").str.split_exact("__", 1).alias("kv"))
        .unnest("kv")
        .rename({"field_0": "column", "field_1": "metric"})
    )
    # pivot to one row per column, metrics as columns
    return long.pivot(index="column", columns="metric", values="value").sort("column")


# ---------- correlations (uses a limited sample to avoid huge collects) ----------


def corr_matrix(
    lf: Frameish,
    method: str = "pearson",
    limit_rows: int = 1_000_000,
) -> pl.DataFrame:
    """
    Correlation matrix over numeric columns. Collects up to `limit_rows`.
    For very large data, raise/lower the limit for speed/memory tradeoff.
    """
    lf = as_lazy(lf)
    num_cols = [c for c, dt in lf.schema.items() if dt in pl.NUMERIC_DTYPES]
    if not num_cols:
        return pl.DataFrame()
    df = lf.select(num_cols).limit(limit_rows).collect()
    # Polars DataFrame.corr returns a (cols x cols) matrix
    return df.corr(method=method)


# ---------- datetime quick features ----------


def datetime_parts(lf: Frameish, col: str) -> pl.DataFrame:
    """Preview calendar components for a datetime column."""
    lf = as_lazy(lf)
    return (
        lf.select(
            pl.col(col).dt.year().alias("year"),
            pl.col(col).dt.month().alias("month"),
            pl.col(col).dt.week().alias("week"),
            pl.col(col).dt.weekday().alias("weekday"),
            pl.col(col).dt.hour().alias("hour"),
        )
        .limit(20)
        .collect()
    )


# ---------- overview ----------


def quick_overview(lf: Frameish, sample_rows: int = 5) -> None:
    lf = as_lazy(lf)
    n, m = shape(lf)
    print(f"Rows: {n:,} | Cols: {m}")
    print("Dtypes:")
    for k, v in lf.schema.items():
        print(f"  - {k}: {v}")
    print("\nHead:")
    print(peek(lf, sample_rows))


# ---------- how to use it -----------

# from polars_eda import *

# # Open (lazy) — accepts CSVs or Parquet paths, or a DataFrame/LazyFrame
# lf = as_lazy([
#     "/Users/sra/Downloads/DE1_0_2008_to_2010_Carrier_Claims_Sample_1A.parquet",
#     "/Users/sra/Downloads/DE1_0_2008_to_2010_Carrier_Claims_Sample_1B.parquet",
# ])

# quick_overview(lf)                 # shape, dtypes, and head
# missingness(lf).head(20)           # top null-heavy columns
# cardinality(lf).head(20)           # highest-cardinality columns
# value_counts(lf, "HCPCS_CD_4", 15) # top codes
# duplicate_rows(lf, subset=["CLM_ID"], k=10)  # most duplicated claim IDs
# numeric_summary(lf)                # stats (one row per numeric column)
# corr_matrix(lf, limit_rows=300_000)  # correlation on a sample


# ---------- putting it all together ----------
def _n_unique_map(lf: pl.LazyFrame) -> dict[str, int]:
    cols = lf.columns
    out = lf.select([pl.col(c).n_unique().alias(c) for c in cols]).collect(
        streaming=True
    )
    return {c: int(out[c][0]) for c in cols}


def _pick_value_counts_col(
    lf: pl.LazyFrame,
    nunique: dict[str, int],
    prefer_patterns=(r"(?i)HCPCS", r"(?i)CPT", r"(?i)ICD"),
    max_card: int = 2000,
) -> Optional[str]:
    # 1) prefer domain columns (HCPCS/CPT/ICD) that are strings
    for pat in prefer_patterns:
        for c, dt in lf.schema.items():
            if re.search(pat, c) and dt == pl.Utf8:
                return c
    # 2) otherwise, first string column with reasonable cardinality
    for c, dt in lf.schema.items():
        if dt == pl.Utf8 and 2 <= nunique.get(c, 0) <= max_card:
            return c
    # 3) fallback: first non-numeric column
    for c, dt in lf.schema.items():
        if dt not in pl.NUMERIC_DTYPES:
            return c
    return None


def _pick_duplicate_key(
    lf: pl.LazyFrame,
    nunique: dict[str, int],
    n_rows: int,
    prefer_exact=("CLM_ID",),
    prefer_patterns=(r"(?i)\bid\b", r"(?i)_id$", r"(?i)\w+ID\b"),
    min_uniqueness: float = 0.5,
) -> list[str]:
    cols = list(lf.schema.keys())
    # 1) exact preferred
    for k in prefer_exact:
        if k in cols:
            return [k]
    # 2) pattern-based IDs with decent uniqueness
    for pat in prefer_patterns:
        for c in cols:
            if re.search(pat, c):
                nu = nunique.get(c, 0)
                if n_rows > 0 and (nu / max(n_rows, 1)) >= min_uniqueness:
                    return [c]
    # 3) fallback: first column
    return [cols[0]] if cols else []


def run_quick_eda(
    x: Union[pl.DataFrame, pl.LazyFrame, str, Path, Iterable[Union[str, Path]]],
    *,
    sample_rows: int = 5,
    missing_top: Optional[int] = 20,
    cardinality_top: Optional[int] = 20,
    value_counts_col: Optional[str] = None,
    value_counts_k: int = 20,
    duplicate_subset: Optional[list[str]] = None,
    duplicate_k: int = 20,
    corr_limit_rows: int = 300_000,
    corr_method: str = "pearson",
    show_columns: bool = True,
) -> None:
    """One-call EDA: prints overview, missingness, cardinality, value counts, dupes, numeric summary, and correlations."""
    lf = as_lazy(x)

    # Column list (before anything else)
    if show_columns:
        print("\n=== Columns ===")
        print(", ".join(lf.columns))

    print("\n=== Quick Overview ===")
    quick_overview(lf)  # prints shape, dtypes, head(sample_rows)
    # Quick Overview already prints head(5) by default; override:
    if sample_rows != 5:
        print(f"\nHead ({sample_rows} rows):")
        print(lf.limit(sample_rows).collect())

    # Precompute counts for heuristics
    n_rows, _ = shape(lf)
    nunique = _n_unique_map(lf)

    # Missingness
    print("\n=== Missingness (top) ===")
    print(missingness(lf, top=missing_top))

    # Cardinality
    print("\n=== Cardinality (n_unique, top) ===")
    print(cardinality(lf, top=cardinality_top))

    # Value counts
    vc_col = value_counts_col or _pick_value_counts_col(lf, nunique)
    if vc_col:
        print(f"\n=== Value Counts: {vc_col} (top {value_counts_k}) ===")
        try:
            print(value_counts(lf, vc_col, k=value_counts_k))
        except Exception as e:
            print(f"[!] value_counts failed for {vc_col}: {e}")
    else:
        print("\n[!] No suitable column found for value counts.")

    # Duplicate rows
    dupe_keys = duplicate_subset or _pick_duplicate_key(lf, nunique, n_rows)
    print(f"\n=== Duplicate Rows by {dupe_keys} (top {duplicate_k}) ===")
    try:
        print(duplicate_rows(lf, subset=dupe_keys, k=duplicate_k))
    except Exception as e:
        print(f"[!] duplicate_rows failed for {dupe_keys}: {e}")

    # Numeric summary
    print("\n=== Numeric Summary ===")
    try:
        print(numeric_summary(lf))
    except Exception as e:
        print(f"[!] numeric_summary failed: {e}")

    # Correlation matrix (sampled)
    print(
        f"\n=== Correlation Matrix (method={corr_method}, limit_rows={corr_limit_rows:,}) ==="
    )
    try:
        cm = corr_matrix(lf, method=corr_method, limit_rows=corr_limit_rows)
        print(cm)
    except Exception as e:
        print(f"[!] corr_matrix failed: {e}")


# Example usage
# from polars_eda import run_quick_eda

# run_quick_eda([
#     "/Users/sra/Downloads/DE1_0_2008_to_2010_Carrier_Claims_Sample_1A.parquet",
#     "/Users/sra/Downloads/DE1_0_2008_to_2010_Carrier_Claims_Sample_1B.parquet",
# ])

# # Override defaults if you want:
# # run_quick_eda(paths, value_counts_col="HCPCS_CD_4", duplicate_subset=["CLM_ID"], corr_limit_rows=200_000)
