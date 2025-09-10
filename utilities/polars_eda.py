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
    # don't rely on alias names; take the first series directly
    counts = lf.select(
        [pl.len()] + [pl.col(c).null_count().alias(c) for c in cols]
    ).collect(streaming=True)
    n = int(counts.to_series(0).item())
    data = (
        pl.DataFrame({"column": cols, "nulls": [int(counts[c][0]) for c in cols]})
        .with_columns((pl.col("nulls") / pl.lit(n) * 100).alias("null_pct"))
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


def value_counts(lf: Frameish, column: str, k: Optional[int] = 20) -> pl.DataFrame:
    """Top-k (or all, if k is None/<=0) value counts for a single column (streaming)."""
    lf = as_lazy(lf)
    q = lf.group_by(pl.col(column)).len().sort(pl.col("len"), descending=True)
    if k and k > 0:
        q = q.limit(k)
    return q.collect(streaming=True).rename({"len": "count"})


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


def duplicate_rows_full(
    lf: Frameish, subset: Optional[List[str]] = None
) -> pl.DataFrame:
    """All duplicates for the subset (no limit). Can be large."""
    lf = as_lazy(lf)
    keys = subset or lf.columns
    return (
        lf.group_by([pl.col(c) for c in keys])
        .len()
        .filter(pl.col("len") > 1)
        .sort(pl.col("len"), descending=True)
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
    method: str = "pearson",  # 'pearson' or 'spearman'
    limit_rows: int = 1_000_000,
    drop_nulls: bool = True,  # pairwise NA handling: simple drop of any-row-with-NA
) -> pl.DataFrame:
    """
    Correlation matrix over numeric columns.
    - Collects up to `limit_rows` rows for speed/memory control.
    - Works on older Polars versions that lack DataFrame.corr(method=...).
    - For 'spearman', ranks each column then computes Pearson on the ranks.
    """
    import numpy as np

    lf = as_lazy(lf)

    # pick numeric columns
    num_cols = [c for c, dt in lf.schema.items() if dt in pl.NUMERIC_DTYPES]
    if not num_cols:
        return pl.DataFrame()

    # sample rows + basic NA handling
    lsel = lf.select(num_cols).limit(limit_rows)
    if drop_nulls:
        lsel = lsel.drop_nulls()
    df = lsel.collect()

    if df.height == 0 or df.width == 0:
        return pl.DataFrame()

    cols = df.columns

    # For Spearman: rank columns first (average rank for ties)
    if method.lower() == "spearman":
        df = df.select([pl.col(c).rank(method="average").alias(c) for c in cols])
    elif method.lower() != "pearson":
        raise ValueError("method must be 'pearson' or 'spearman'")

    # convert to numpy, remove zero-variance columns to avoid NaNs
    arr = df.to_numpy()
    std = np.nanstd(arr, axis=0)
    keep = std > 0
    kept_cols = [c for c, k in zip(cols, keep) if k]
    if not any(keep):
        # no variance anywhere -> return empty
        return pl.DataFrame()

    arr = arr[:, keep]

    # compute correlation
    cm = np.corrcoef(arr, rowvar=False)

    # build a friendly DataFrame: first column is the row label, then one col per variable
    out = pl.DataFrame(cm, schema=kept_cols)
    out = pl.concat([pl.DataFrame({"column": kept_cols}), out], how="horizontal")
    return out


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


# ---------- artifact utils (tables and plots) -----------


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_table(df: pl.DataFrame, out_path: Path, table_format: str = "csv") -> Path:
    """Save a Polars DataFrame as CSV or Parquet (default CSV)."""
    out_path = out_path.with_suffix("." + table_format.lower())
    _ensure_dir(out_path.parent)
    if table_format.lower() == "parquet":
        df.write_parquet(out_path)
    else:
        df.write_csv(out_path)
    return out_path


def _maybe_import_seaborn():
    try:
        import seaborn as sns  # noqa: F401

        return True
    except Exception:
        return False


def plot_corr_lower_triangle(
    cm_df: pl.DataFrame,
    out_png: Path,
    title: str = "Correlation",
    cmap: str = "coolwarm",
    dpi: int = 200,
) -> Path:
    """
    Draw a lower-triangle heatmap from the correlation matrix returned by corr_matrix().
    Expects a frame with a 'column' label column and symmetric numeric columns.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    _ensure_dir(out_png.parent)
    pdf = cm_df.to_pandas()
    pdf = pdf.set_index("column")
    data = pdf.loc[:, pdf.columns]  # square

    # mask upper triangle
    mask = np.triu(np.ones_like(data, dtype=bool))

    use_sns = _maybe_import_seaborn()
    plt.figure(figsize=(max(8, len(data) * 0.35), max(6, len(data) * 0.35)))
    if use_sns:
        import seaborn as sns

        ax = sns.heatmap(
            data,
            mask=mask,
            cmap=cmap,
            vmin=-1,
            vmax=1,
            square=True,
            cbar=True,
            linewidths=0.0,
        )
    else:
        # fallback to plain matplotlib
        mdata = data.mask(mask)
        plt.imshow(mdata, vmin=-1, vmax=1, cmap=cmap)
        plt.colorbar()
        plt.xticks(range(len(data.columns)), data.columns, rotation=90)
        plt.yticks(range(len(data.index)), data.index)
        ax = plt.gca()

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    plt.close()
    return out_png


def plot_topk_bar(
    df: pl.DataFrame,
    x: str,
    y: str,
    out_png: Path,
    title: str,
    horizontal: bool = True,
    dpi: int = 200,
) -> Path:
    """
    Simple bar plot for top-k value counts or other summaries.
    Uses seaborn if available, else matplotlib.
    """
    import matplotlib.pyplot as plt

    _ensure_dir(out_png.parent)
    pdf = df.to_pandas()

    use_sns = _maybe_import_seaborn()
    plt.figure(figsize=(10, max(4, min(0.35 * len(pdf), 16))))
    if horizontal:
        if use_sns:
            import seaborn as sns

            sns.barplot(data=pdf, x=y, y=x)
        else:
            plt.barh(pdf[x], pdf[y])
        plt.xlabel(y)
        plt.ylabel(x)
    else:
        if use_sns:
            import seaborn as sns

            sns.barplot(data=pdf, x=x, y=y)
        else:
            plt.bar(pdf[x], pdf[y])
        plt.ylabel(y)
        plt.xticks(rotation=90)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    plt.close()
    return out_png


def run_quick_eda(
    x: Union[pl.DataFrame, pl.LazyFrame, str, Path, Iterable[Union[str, Path]]],
    *,
    sample_rows: int = 5,
    missing_top: Optional[int] = 20,
    cardinality_top: Optional[int] = 20,
    value_counts_col: Optional[str] = None,
    value_counts_k: Optional[int] = 20,  # now Optional: None => all
    duplicate_subset: Optional[list[str]] = None,
    duplicate_k: int = 20,
    corr_limit_rows: int = 300_000,
    corr_method: str = "pearson",
    show_columns: bool = True,
    artifacts_dir: Optional[
        Union[str, Path]
    ] = None,  # save full tables & plots here if set
    table_format: str = "csv",  # "csv" or "parquet"
    make_plots: bool = True,  # also emit PNGs when artifacts_dir is set
    plot_dpi: int = 200,
) -> None:
    """One-call EDA: prints overview, missingness, cardinality, value counts, dupes, numeric summary, and correlations."""
    lf = as_lazy(x)
    outdir = Path(artifacts_dir) if artifacts_dir else None
    if outdir:
        _ensure_dir(outdir)

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
    miss_df = missingness(lf, top=missing_top)
    print(miss_df)
    if outdir:
        # full table (not top-restricted)
        miss_full = missingness(lf, top=None)
        _save_table(miss_full, outdir / "missingness", table_format)

    # Cardinality
    print("\n=== Cardinality (n_unique, top) ===")
    card_df = cardinality(lf, top=cardinality_top)
    print(card_df)
    if outdir:
        card_full = cardinality(lf, top=None)
        _save_table(card_full, outdir / "cardinality", table_format)

    # Value counts
    vc_col = value_counts_col or _pick_value_counts_col(lf, nunique)
    if vc_col:
        print(
            f"\n=== Value Counts: {vc_col} (top {value_counts_k if value_counts_k else 'all'}) ==="
        )
        vc_top_df = value_counts(lf, vc_col, k=value_counts_k)
        print(vc_top_df)
        if outdir:
            vc_all_df = value_counts(lf, vc_col, k=None)
            _save_table(vc_all_df, outdir / f"value_counts__{vc_col}", table_format)
            if make_plots:
                plot_topk_bar(
                    vc_top_df,
                    x=vc_col,
                    y="count",
                    out_png=outdir / f"value_counts__{vc_col}.png",
                    title=f"Top value counts: {vc_col}",
                    horizontal=True,
                    dpi=plot_dpi,
                )
    else:
        print("\n[!] No suitable column found for value counts.")

    # Duplicate rows
    dupe_keys = duplicate_subset or _pick_duplicate_key(lf, nunique, n_rows)
    print(f"\n=== Duplicate Rows by {dupe_keys} (top {duplicate_k}) ===")
    try:
        dupe_top_df = duplicate_rows(lf, subset=dupe_keys, k=duplicate_k)
        print(dupe_top_df)
        if outdir:
            dupe_full_df = duplicate_rows_full(lf, subset=dupe_keys)
            _save_table(
                dupe_full_df,
                outdir / f"duplicates__{'__'.join(dupe_keys)}",
                table_format,
            )
            if make_plots and dupe_top_df.height > 0:
                plot_topk_bar(
                    dupe_top_df.rename({dupe_keys[0]: "key"}),
                    x="key",
                    y="dupe_count",
                    out_png=outdir / f"duplicates__{'__'.join(dupe_keys)}.png",
                    title=f"Top duplicates by {', '.join(dupe_keys)}",
                    horizontal=True,
                    dpi=plot_dpi,
                )
    except Exception as e:
        print(f"[!] duplicate_rows failed for {dupe_keys}: {e}")

    # Numeric summary
    print("\n=== Numeric Summary ===")
    try:
        numsum_df = numeric_summary(lf)
        print(numsum_df)
        if outdir:
            _save_table(numsum_df, outdir / "numeric_summary", table_format)
    except Exception as e:
        print(f"[!] numeric_summary failed: {e}")

    # Correlation matrix (sampled)
    print(
        f"\n=== Correlation Matrix (method={corr_method}, limit_rows={corr_limit_rows:,}) ==="
    )
    try:
        cm = corr_matrix(lf, method=corr_method, limit_rows=corr_limit_rows)
        print(cm)
        if outdir and cm.height > 0 and "column" in cm.columns:
            _save_table(cm, outdir / "correlation_matrix", table_format)
            if make_plots:
                plot_corr_lower_triangle(
                    cm,
                    out_png=outdir / "correlation_matrix.png",
                    title=f"Correlation ({corr_method})",
                    dpi=plot_dpi,
                )
    except Exception as e:
        print(f"[!] corr_matrix failed: {e}")


# Example usage
# from utilities.polars_eda import run_quick_eda

# run_quick_eda(
#     [
#         "../data/sample_data/DE1_0_2008_to_2010_Carrier_Claims_Sample_1A.parquet",
#         "../data/sample_data/DE1_0_2008_to_2010_Carrier_Claims_Sample_1B.parquet",
#     ],
#     artifacts_dir="../eda_artifacts",  # folder will be created
#     table_format="csv",                # or "parquet"
#     make_plots=True,                   # emit PNGs alongside tables
#     value_counts_k=30,                 # show top-30 in console/plot, save ALL to CSV
#     corr_limit_rows=300_000,           # you already had this
# )
#
# outputs:
#
# ../eda_artifacts/
#   missingness.csv
#   cardinality.csv
#   value_counts__HCPCS_CD_4.csv
#   value_counts__HCPCS_CD_4.png
#   duplicates__CLM_ID.csv
#   duplicates__CLM_ID.png
#   numeric_summary.csv
#   correlation_matrix.csv
#   correlation_matrix.png   <-- lower-triangle heatmap (blue↔red)
