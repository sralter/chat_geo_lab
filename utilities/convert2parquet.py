#!/usr/bin/env python3
import argparse
from pathlib import Path
import re
import sys
import csv

try:
    import polars as pl
except ImportError:
    sys.stderr.write(
        "Polars is not installed. Install with: python -m pip install -U polars\n"
    )
    sys.exit(1)


# --- header peeking without type inference ---
def read_header_csv_only(path: Path):
    with open(path, "r", newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        row = next(reader, None)
        if row is None:
            raise ValueError(f"No header row found in {path}")
        return row


def build_schema_overrides(csv_paths, patterns, extra_names, print_overrides=False):
    compiled = [re.compile(p) for p in patterns]
    overrides = {}
    for p in csv_paths:
        p = Path(p).expanduser().resolve()
        try:
            cols = read_header_csv_only(p)
        except Exception as e:
            print(f"[!] Could not read header for {p}: {e}", file=sys.stderr)
            continue
        for c in cols:
            if c in extra_names or any(rx.search(c) for rx in compiled):
                overrides[c] = pl.Utf8
    if print_overrides:
        if overrides:
            print("[i] Forcing Utf8 on columns:", ", ".join(sorted(overrides.keys())))
        else:
            print("[i] No columns matched for Utf8 overrides.")
    return overrides


# --- scanning helper (handles older/newer Polars kwarg names) ---
def scan_csv_with_overrides(
    path_or_paths, infer_schema_length, overrides, ignore_errors
):
    kwargs = dict(infer_schema_length=infer_schema_length)
    if overrides:
        try:
            kwargs["schema_overrides"] = overrides
            return pl.scan_csv(path_or_paths, ignore_errors=ignore_errors, **kwargs)
        except TypeError:
            kwargs.pop("schema_overrides", None)
            kwargs["dtypes"] = overrides
    try:
        return pl.scan_csv(path_or_paths, ignore_errors=ignore_errors, **kwargs)
    except TypeError:
        kwargs.pop("ignore_errors", None)
        return pl.scan_csv(path_or_paths, **kwargs)


# --- verification: re-open parquet and print first N rows ---
def verify_parquet(path: Path, n_rows: int = 1):
    try:
        df = pl.read_parquet(str(path), n_rows=n_rows)
        print(f"[✓] Verified {path} — first {n_rows} row(s):")
        print(df)  # pretty table
    except Exception as e:
        print(f"[x] Could not verify {path}: {e}", file=sys.stderr)


# --- write helpers ---
def to_parquet_each(
    csv_paths,
    infer_schema_length,
    compression,
    select,
    overrides,
    ignore_errors,
    verify_rows,
):
    for f in csv_paths:
        src = Path(f).expanduser().resolve()
        if not src.exists():
            print(f"[!] Missing: {src}", file=sys.stderr)
            continue
        dst = src.with_suffix(".parquet")
        lf = scan_csv_with_overrides(
            str(src), infer_schema_length, overrides, ignore_errors
        )
        if select:
            lf = lf.select(select)
        lf.sink_parquet(str(dst), compression=compression, statistics=True)
        print(f"[✓] Wrote {dst}")
        if verify_rows > 0:
            verify_parquet(dst, verify_rows)


def to_parquet_merged(
    csv_paths,
    out_path,
    infer_schema_length,
    compression,
    select,
    overrides,
    ignore_errors,
    verify_rows,
):
    out = Path(out_path).expanduser().resolve()
    paths = [str(Path(p).expanduser().resolve()) for p in csv_paths]
    lf = scan_csv_with_overrides(paths, infer_schema_length, overrides, ignore_errors)
    if select:
        lf = lf.select(select)
    lf.sink_parquet(str(out), compression=compression, statistics=True)
    print(f"[✓] Wrote {out}")
    if verify_rows > 0:
        verify_parquet(out, verify_rows)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Fast CSV→Parquet with Polars (streaming). "
            "By default writes one Parquet per CSV; with -o merges all inputs."
        )
    )
    ap.add_argument("csv", nargs="+", help="Path(s) to CSV file(s).")
    ap.add_argument("-o", "--out", help="Output Parquet path for merged output.")
    ap.add_argument(
        "--compression",
        choices=["snappy", "zstd"],
        default="snappy",
        help="Parquet compression (default: snappy).",
    )
    ap.add_argument(
        "--infer-schema-length",
        type=int,
        default=10000,
        help="Rows to sample for schema inference (default: 10000).",
    )
    ap.add_argument(
        "--select", nargs="+", help="Optional list of columns to keep (faster/smaller)."
    )
    ap.add_argument(
        "--strings-for",
        nargs="*",
        default=[r"(?i)ICD9", r"(?i)HCPCS", r"(?i)CPT"],
        help="Regex patterns for columns to force as strings (Utf8).",
    )
    ap.add_argument(
        "--strings-cols",
        nargs="*",
        default=[],
        help="Exact column names to force as strings (Utf8).",
    )
    ap.add_argument(
        "--loose",
        action="store_true",
        help="Allow CSV reader to ignore row-level parse errors.",
    )
    ap.add_argument(
        "--print-overrides",
        action="store_true",
        help="Print which columns are being forced to Utf8.",
    )
    ap.add_argument(
        "--verify-rows",
        type=int,
        default=1,
        help="After writing, open each Parquet and print first N rows (0 to skip).",
    )
    args = ap.parse_args()

    overrides = build_schema_overrides(
        args.csv,
        args.strings_for,
        args.strings_cols,
        print_overrides=args.print_overrides,
    )
    ignore_errors = bool(args.loose)

    if args.out:
        to_parquet_merged(
            args.csv,
            args.out,
            args.infer_schema_length,
            args.compression,
            args.select,
            overrides,
            ignore_errors,
            args.verify_rows,
        )
    else:
        to_parquet_each(
            args.csv,
            args.infer_schema_length,
            args.compression,
            args.select,
            overrides,
            ignore_errors,
            args.verify_rows,
        )


if __name__ == "__main__":
    main()
