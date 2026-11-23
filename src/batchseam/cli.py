import argparse
from contextlib import redirect_stdout
from importlib import import_module
import json
from pathlib import Path
import runpy
import sys

import polars as pl

from .core import check_batches
from .repro import load_repro, save_repro


def _cuts(text):
    try:
        return tuple(int(value) for value in text.split(","))
    except ValueError as err:
        raise argparse.ArgumentTypeError("use offsets such as 2 or 2,5") from err


def _load_transform(spec):
    filename, sep, name = spec.rpartition(":")
    if not sep or not filename or not name.isidentifier():
        raise ValueError("use transform.py:function")
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
    if not callable(transform):
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


_DEMOS = {"running-total": _running_total, "sensor": _sensor}


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
    if argv is None:
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description="Find Polars transforms that change at batch boundaries.")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="try a transform that changes at a batch boundary")
    demo.add_argument("example", nargs="?", choices=_DEMOS, default="running-total")
    check = commands.add_parser("check", help="test a local DataFrame transform against a CSV or Parquet fixture")
    check.add_argument("transform", metavar="FILE.py:FUNCTION")
    check.add_argument("input", type=Path)
    replay = commands.add_parser("replay", help="check the transform and split from a saved repro")
    replay.add_argument("directory", metavar="DIR", type=Path)
    for command in (demo, check):
        group = command.add_mutually_exclusive_group()
        group.add_argument("--cuts", type=_cuts, help="replay a split at these zero-based offsets")
        group.add_argument("--max-plans", type=int, default=16, help="maximum number of candidate splits (default 16)")
        group.add_argument("--exhaustive", action="store_true", help="check every single interior cut until a difference")
        command.add_argument("--save-repro", metavar="DIR", type=Path, help="save a reduced fixture and regression test on failure")
    for command in (demo, check, replay):
        command.add_argument("--json", action="store_true", help="emit the result as JSON on stdout")
    args = parser.parse_args(argv)
    original_path = sys.path[:]
    try:
        with redirect_stdout(sys.stderr):
            if args.command == "replay":
                spec, frame, cuts = load_repro(args.directory)
                if spec.startswith("demo:"):
                    transform = _DEMOS[spec.removeprefix("demo:")]
                else:
                    transform, _ = _load_transform(spec)
                report = check_batches(transform, frame, cuts=cuts)
            else:
                if args.command == "demo":
                    spec, transform = f"demo:{args.example}", _DEMOS[args.example]
                    if args.example == "sensor":
                        readings = list(range(100))
                        readings[41] = None
                        frame = pl.DataFrame({"reading": readings})
                    else:
                        frame = pl.DataFrame({"amount": [8, 3, 4, -2]})
                else:
                    transform, spec = _load_transform(args.transform)
                    frame = _read_frame(args.input)
                report = check_batches(transform, frame, cuts=args.cuts, max_plans=args.max_plans,
                                       exhaustive=args.exhaustive)
                if args.save_repro is not None and report["status"] == "fail":
                    report["repro"] = save_repro(transform, frame, report, args.save_repro, spec=spec)
    except Exception as err:
        parser.exit(2, f"batchseam: {err}\n")
    finally:
        sys.path[:] = original_path
    if args.json:
        print(json.dumps(report, allow_nan=False))
    else:
        _print_report(report)
        if "repro" in report:
            print(f"Saved repro to {args.save_repro}")
    return {"pass": 0, "fail": 1, "insufficient": 2}[report["status"]]
