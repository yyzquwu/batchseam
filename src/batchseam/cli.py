import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import runpy
import sys

import polars as pl

from .core import check_batches


def _cuts(text):
    try:
        return tuple(int(value) for value in text.split(","))
    except ValueError as err:
        raise argparse.ArgumentTypeError("use offsets such as 2 or 2,5") from err


def _inputs(spec, path):
    filename, sep, name = spec.rpartition(":")
    if not sep or not filename or not name.isidentifier():
        raise ValueError("use transform.py:function")
    script = Path(filename).resolve(strict=True)
    # Make sibling modules importable just as when running the script directly.
    sys.path.insert(0, str(script.parent))
    transform = runpy.run_path(str(script)).get(name)
    if not callable(transform):
        raise ValueError(f"Could not find callable {name!r} in {script.name}")
    readers = {".parquet": pl.read_parquet, ".csv": pl.read_csv}
    if path.suffix.lower() not in readers:
        raise ValueError("input file must be CSV or Parquet")
    return transform, readers[path.suffix.lower()](path)


def _running_total(df):
    return df.with_columns(balance=pl.col("amount").cum_sum())


def _print_report(report):
    plans = "plan" if report["checked"] == 1 else "plans"
    print(f"{report['status'].upper()}  |  {report['input_rows']} input rows  |  {report['checked']} split {plans} checked")
    if report["status"] == "insufficient":
        print("At least two rows are needed to split the input.")
    if report["status"] != "fail":
        return
    diff = report["difference"]
    print("Replay this split with --cuts", ",".join(map(str, report["cuts"])))
    if diff["kind"] == "value":
        print(f"First difference at output row {diff['output_row']} (zero-based), column {diff['column']!r}")
    elif diff["kind"] == "row_count":
        print("Different output row counts")
    else:
        start, end = diff["input_span"]
        print(f"{diff['kind']} on input rows [{start}, {end})")
    if "whole" in diff:
        print("Whole frame:", json.dumps(diff["whole"]))
        print("Batched:    ", json.dumps(diff["batched"]))
    if "error" in diff:
        print(diff["error"])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Find Polars transforms that change at batch boundaries.")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="try a running total that resets on every batch")
    check = commands.add_parser("check", help="test a local DataFrame transform against a CSV or Parquet fixture")
    check.add_argument("transform", metavar="FILE.py:FUNCTION")
    check.add_argument("input", type=Path)
    for command in (demo, check):
        group = command.add_mutually_exclusive_group()
        group.add_argument("--cuts", type=_cuts, help="replay a split at these zero-based offsets")
        group.add_argument("--max-plans", type=int, default=16, help="maximum number of candidate splits (default 16)")
        command.add_argument("--json", action="store_true", help="emit the result as JSON on stdout")
    args = parser.parse_args(argv)
    original_path = sys.path[:]
    try:
        with redirect_stdout(sys.stderr):
            if args.command == "demo":
                transform, frame = _running_total, pl.DataFrame({"amount": [8, 3, 4, -2]})
            else:
                transform, frame = _inputs(args.transform, args.input)
            report = check_batches(transform, frame, cuts=args.cuts, max_plans=args.max_plans)
    except Exception as err:
        parser.exit(2, f"batchseam: {err}\n")
    finally:
        sys.path[:] = original_path
    if args.json:
        print(json.dumps(report, allow_nan=False))
    else:
        _print_report(report)
    return {"pass": 0, "fail": 1, "insufficient": 2}[report["status"]]
