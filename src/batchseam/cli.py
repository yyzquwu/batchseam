import argparse
from contextlib import contextmanager, nullcontext, redirect_stdout
from importlib import import_module
import json
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory

import polars as pl

from .core import _DEFAULT_MAX_PLANS, check_batches, check_elementwise, check_expr_elementwise
from .repro import load_repro_case, save_repro


@contextmanager
def _current_source():
    """Load current source even when an edit has the same timestamp and size as its pyc."""
    prefix, disabled = sys.pycache_prefix, sys.dont_write_bytecode
    with TemporaryDirectory() as empty:
        try:
            sys.pycache_prefix, sys.dont_write_bytecode = empty, True
            yield
        finally:
            sys.pycache_prefix, sys.dont_write_bytecode = prefix, disabled


def _cuts(text):
    try:
        return tuple(int(value) for value in text.split(","))
    except ValueError as err:
        raise argparse.ArgumentTypeError("use offsets such as 2 or 2,5") from err


def _load_transform(spec, *, expr=False):
    filename, sep, name = spec.rpartition(":")
    if not sep or not filename or not name.isidentifier():
        raise ValueError("use FILE.py:NAME for an expression target" if expr else "use transform.py:function")
    script = Path(filename).resolve(strict=True)
    root = script.parent
    while root != root.parent and (root / "__init__.py").is_file():
        root = root.parent
    sys.path[:0] = list(dict.fromkeys(map(str, (root, script.parent, Path.cwd()))))
    if root == script.parent:
        namespace = runpy.run_path(str(script))
    elif script.suffix == ".py" and script.stem.isidentifier():
        parts = script.relative_to(root).with_suffix("").parts
        module = import_module(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
        if Path(module.__file__).resolve() != script:
            raise ValueError(f"Imported {module.__name__} from {module.__file__}, expected {script}")
        namespace = vars(module)
    else:
        # A filename like daily.v1.py is one file, not two levels of package imports.
        package = ".".join(script.parent.relative_to(root).parts)
        namespace = runpy.run_path(str(script), run_name=f"{package}.__batchseam_transform__")
    transform = namespace.get(name)
    if expr:
        if callable(transform) and not isinstance(transform, pl.Expr):
            try:
                transform = transform()
            except Exception as err:
                raise ValueError(f"Expression factory {name!r} failed when called with no arguments: {err}") from err
            if not isinstance(transform, pl.Expr):
                raise ValueError(f"Expression factory {name!r} must return a Polars Expr")
        elif not isinstance(transform, pl.Expr):
            raise ValueError(f"Expected a Polars Expr or zero-argument factory named {name!r} in {script.name}")
    elif isinstance(transform, pl.Expr):
        raise ValueError(f"{name!r} is a Polars expression. Use --expr to check it.")
    elif not callable(transform):
        raise ValueError(f"Could not find callable {name!r} in {script.name}")
    return transform, f"{script}:{name}"


def _read_frame(path):
    readers = {".parquet": pl.read_parquet, ".csv": pl.read_csv}
    if path.suffix.lower() not in readers:
        raise ValueError("input file must be CSV or Parquet")
    return readers[path.suffix.lower()](path)


def _running_total(df):
    return df.with_columns(balance=pl.col("amount").cum_sum())


def _sensor(df):
    return df.with_columns(pl.col("reading").forward_fill())


def _dates(series):
    return series.str.to_date("%Y-%m-%d")


def _center(series):
    return series - series.mean()


def _positive(series):
    return series.filter(series > 0)


def _first_rate(columns):
    amount, rate = columns
    return amount * rate.first()


_DEMOS = {
    "running-total": (_running_total, None, {"amount": [8, 3, 4, -2]}),
    "sensor": (_sensor, None, {"reading": [*range(41), None, *range(42, 100)]}),
    "dates": (_dates, "date", {"date": ["2026-11-20", None, "2026-11-22"]}),
    "center": (_center, "amount", {"amount": [10, 30]}),
    "positive": (_positive, "amount", {"amount": [-1, 2, 3]}),
    "fx": (pl.map_batches(["amount", "rate"], _first_rate, return_dtype=pl.Float64,
                          is_elementwise=True).alias("usd"), None, {"amount": [10, 10], "rate": [1.0, 2.0]}),
}


def _check(transform, frame, series, expr, **options):
    if expr:
        return check_expr_elementwise(transform, frame, **options)
    if series is None:
        return check_batches(transform, frame, **options)
    return check_elementwise(transform, frame.get_column(series), **options)


def _print_report(report, series, expr):
    plans = "plan" if report["checked"] == 1 else "plans"
    print(f"{report['status'].upper()}  |  {report['input_rows']} input rows  |  {report['checked']} split {plans} checked")
    if "coverage" in report:
        coverage = report["coverage"]
        column = f"  |  column {coverage['column']!r}" if "column" in coverage else ""
        print(f"Fixture: {coverage['fixture']}{column}")
        if "plans" in coverage:
            print("Tested splits:", json.dumps(coverage["plans"]))
        else:
            first, last = coverage["single_cut_range"]
            print(f"Tested every single cut from {first} through {last}.")
    if report["status"] == "insufficient":
        print("At least two rows are needed to split the input.")
    if report["status"] != "fail":
        return
    diff = report["difference"]
    if report["cuts"]:
        print("Replay this split with --cuts", ",".join(map(str, report["cuts"])))
    if diff["kind"] == "value":
        print(f"First difference at output row {diff['output_row']} (zero-based), column {diff['column']!r}")
    elif diff["kind"] == "row_count":
        print("Different output row counts")
    elif diff["kind"] == "length":
        scope = "Whole expression" if expr else "Whole Series"
        if diff["scope"] == "batch":
            start, end = diff["input_span"]
            batch = "Expression batch" if expr else "Batch"
            scope = f"{batch} on input rows [{start}, {end})"
        print(f"{scope} length mismatch (expected {diff['expected']}, actual {diff['actual']})")
    else:
        start, end = diff["input_span"]
        print(f"{diff['kind']} on input rows [{start}, {end})")
    if "whole" in diff:
        mode = "expression" if expr else "frame" if series is None else "Series"
        print(f"Whole {mode}:", json.dumps(diff["whole"]))
        print("Batched:    ", json.dumps(diff["batched"]))
    if "input_values" in diff:
        print("Input row values:", json.dumps(diff["input_values"]))
    if "error" in diff:
        print(diff["error"])


def main(argv=None):
    if argv is None:
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description="Find Polars transforms that change at batch boundaries.")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="try a built-in DataFrame, Series, or expression check")
    demo.add_argument("example", nargs="?", choices=_DEMOS, default="running-total")
    check = commands.add_parser("check", help="test a local transform against a CSV or Parquet fixture")
    check.add_argument("transform", metavar="FILE.py:NAME")
    check.add_argument("input", type=Path)
    mode = check.add_mutually_exclusive_group()
    mode.add_argument("--series", metavar="COLUMN", help="check a Series callback on this input column")
    mode.add_argument("--expr", action="store_true", help="check a Polars expression or zero-argument expression factory")
    replay = commands.add_parser("replay", help="check the transform and split from a saved repro")
    replay.add_argument("directory", metavar="DIR", type=Path)
    scan = commands.add_parser("scan", help="find batch claims in Python source without running it")
    scan.add_argument("path", nargs="?", default=Path("."), type=Path, metavar="PATH")
    suite = commands.add_parser("suite", help="run named project checks from a TOML manifest")
    suite.add_argument("manifest", nargs="?", default=Path("batchseam.toml"), type=Path)
    for command in (demo, check):
        group = command.add_mutually_exclusive_group()
        group.add_argument("--cuts", type=_cuts, help="replay a split at these zero-based offsets")
        group.add_argument("--max-plans", type=int, help="maximum number of candidate splits (default 16)")
        group.add_argument("--exhaustive", action="store_true", help="check every single interior cut until a difference")
        command.add_argument("--save-repro", metavar="DIR", type=Path, help="save a reduced fixture and regression test on failure")
    for command in (demo, check, replay, scan, suite):
        command.add_argument("--json", action="store_true", help="emit the result as JSON on stdout")
    args = parser.parse_args(argv)
    if args.command == "suite":
        from .suite import run_suite

        try:
            result = run_suite(args.manifest)
        except (OSError, ValueError) as err:
            parser.exit(2, f"batchseam: {err}\n")
        if args.json:
            print(json.dumps(result, allow_nan=False))
        else:
            for item in result["checks"]:
                print(f"{item['status'].upper():12}{item['name']!r}")
                if "diagnostic" in item:
                    print(item["diagnostic"])
                elif item["status"] == "fail":
                    print("  " + json.dumps(item["report"]["difference"], ensure_ascii=False))
            print(f"Suite {result['status'].upper()} | {len(result['checks'])} checks")
        return {"pass": 0, "fail": 1, "insufficient": 2, "error": 2}[result["status"]]
    if args.command == "scan":
        from .scan_cli import scan_report

        return scan_report(args.path, as_json=args.json)
    original_path = sys.path[:]
    try:
        with redirect_stdout(sys.stderr), (_current_source() if args.command in ("check", "replay") else nullcontext()):
            if args.command == "replay":
                fixture_label = str(args.directory / "fixture.parquet")
                spec, frame, cuts, series, expr = load_repro_case(args.directory)
                if spec.startswith("demo:"):
                    transform = _DEMOS[spec.removeprefix("demo:")][0]
                else:
                    transform, _ = _load_transform(spec, expr=expr)
                report = _check(transform, frame, series, expr, cuts=cuts or None)
            else:
                if args.command == "demo":
                    spec = f"demo:{args.example}"
                    fixture_label = spec
                    transform, series, data = _DEMOS[args.example]
                    expr = isinstance(transform, pl.Expr)
                    frame = pl.DataFrame(data)
                else:
                    fixture_label = str(args.input)
                    expr = args.expr
                    transform, spec = _load_transform(args.transform, expr=expr)
                    frame = _read_frame(args.input)
                    series = args.series
                    if series is not None:
                        if series not in frame.columns:
                            raise ValueError(f"Series column {series!r} was not found in the input")
                        frame = frame.get_column(series).to_frame()
                budget = _DEFAULT_MAX_PLANS if args.max_plans is None else args.max_plans
                report = _check(transform, frame, series, expr, cuts=args.cuts, max_plans=budget,
                                exhaustive=args.exhaustive)
                if args.save_repro is not None and report["status"] == "fail":
                    report["repro"] = save_repro(transform, frame, report, args.save_repro,
                                                 spec=spec, series=series, expr=expr)
    except Exception as err:
        parser.exit(2, f"batchseam: {err}\n")
    finally:
        sys.path[:] = original_path
    if "coverage" in report:
        report["coverage"]["fixture"] = fixture_label
    if args.json:
        print(json.dumps(report, allow_nan=False))
    else:
        _print_report(report, series, expr)
        if "repro" in report:
            print(f"Saved repro to {args.save_repro}")
    return {"pass": 0, "fail": 1, "insufficient": 2}[report["status"]]
