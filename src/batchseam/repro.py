"""Save a reduced, verified fixture for a failing batch check."""

from contextlib import ExitStack
import json
from os.path import relpath
from pathlib import Path
from uuid import uuid4

import polars as pl
from polars.testing import assert_frame_equal

from . import core

_DEMOS = {"demo:sensor", "demo:running-total", "demo:dates", "demo:center", "demo:positive"}
_EXPR_DEMOS = {"demo:fx"}
_MAX_ATTEMPTS = 128
_FRAME_VERSION, _SERIES_VERSION, _EXPR_VERSION = 1, 2, 3
_TEST = '''import json
from pathlib import Path
import os
import subprocess
import sys


def test_batchseam_regression():
    # Start the installed CLI first, then restore this project's pytest import paths.
    runner = (
        "import json, sys\\n"
        "from batchseam.cli import main\\n"
        "sys.path[:0] = json.load(sys.stdin)\\n"
        "raise SystemExit(main(['replay', sys.argv[1], '--json']))"
    )
    result = subprocess.run(
        [sys.executable, "-P", "-c", runner, str(Path(__file__).resolve().parent)],
        env={k: v for k, v in os.environ.items() if k.upper() != "PYTHONPATH"},
        input=json.dumps([os.path.abspath(p) for p in sys.path]),
        capture_output=True, text=True,
    )
    expected_code, expected_status = _EXPECTED
    assert (result.returncode == expected_code and result.stdout
            and json.loads(result.stdout)["status"] == expected_status), (
        f"batchseam replay exited with {result.returncode}\\n{result.stdout}{result.stderr}"
    )
'''


def _file_spec(spec):
    filename, sep, name = spec.rpartition(":")
    if not sep or not filename or not name.isidentifier():
        raise ValueError("Expected a transform reference of the form FILE.py:function")
    return Path(filename), name


def _bundle_path(path, dest):
    try:
        return Path(relpath(path, dest.resolve())).as_posix()
    except ValueError:  # Windows paths on different drives have no relative path.
        return str(path)


def _saved_spec(spec, dest, *, expr):
    if spec in (_EXPR_DEMOS if expr else _DEMOS):
        return spec
    source, name = _file_spec(spec)
    if not source.is_absolute():
        raise ValueError("Saving a repro requires an absolute transform path")
    if expr and not source.is_file():
        raise ValueError(f"Saving an expression repro requires an existing source file: {source}")
    return f"{_bundle_path(source, dest)}:{name}"


def _signature(report):
    if report["status"] != "fail":
        return None
    diff = report["difference"]
    kind = diff["kind"]
    if kind == "value":
        return kind, diff["column"]
    if kind == "batch_error":
        return kind, diff["error"]
    if kind == "row_count":
        return kind, diff["whole"] > diff["batched"]
    if kind == "length":
        return kind, diff["scope"], diff["actual"] > diff["expected"]
    return kind, diff["whole"], diff["batched"]


def _reduce(run_check, frame, report):
    # Try progressively finer contiguous deletions within a fixed candidate budget.
    signature = _signature(report)
    whole_failure = not report["cuts"]
    groups, attempts = 2, 0
    while frame.height > 2:
        rows = frame.height
        for i in range(groups):
            if attempts == _MAX_ATTEMPTS:
                return frame, report
            attempts += 1
            start, end = i * rows // groups, (i + 1) * rows // groups

            def remap(offset):
                return offset - max(0, min(offset, end) - start)

            # Drop edges and duplicate offsets from batches with no remaining rows.
            size = rows - (end - start)
            cuts = list(dict.fromkeys(cut for cut in map(remap, report["cuts"]) if 0 < cut < size))
            if size < 2 or (not cuts and not whole_failure):
                continue
            span = list(map(remap, report["difference"].get("input_span", [])))
            candidate = pl.concat([frame.slice(0, start), frame.slice(end)])
            try:
                check = run_check(candidate, cuts)
            except Exception:
                continue
            if _signature(check) == signature and check["difference"].get("input_span", []) == span:
                frame, report = candidate, check
                groups = min(size, max(2, groups - 1))
                break
        else:
            if groups == rows:
                break
            groups = min(rows, groups * 2)
    return frame, report


def save_repro(transform, frame, report, dest, *, spec, series=None, expr=False, workdir=None) -> dict:
    """Reduce a failure and verify its Parquet fixture.

    For expr=True, pass a Polars Expr and a source reference that can rebuild it.
    """
    if expr and series is not None:
        raise ValueError("An expression repro cannot select a Series column")
    if expr and not isinstance(transform, pl.Expr):
        raise TypeError("Saving an expression repro requires a Polars Expr")
    if report["status"] != "fail" or report["input_rows"] != frame.height:
        raise ValueError("Saving a repro requires a failing report for this fixture")
    dest = Path(dest).absolute()
    ref = _saved_spec(spec, dest, expr=expr)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError(f"Repro destination already exists: {dest}")
    if not dest.parent.is_dir():
        raise NotADirectoryError(f"Choose an existing parent directory for the repro: {dest.parent}")
    if series is not None:
        if not isinstance(series, str) or series not in frame.columns:
            raise ValueError("Choose a Series column from the input fixture")
        frame = frame.get_column(series).to_frame()

    def run_check(fixture, cuts):
        if expr:
            return core.check_expr_elementwise(transform, fixture, cuts=cuts or None)
        if series is None:
            return core.check_batches(transform, fixture, cuts=cuts)
        return core.check_elementwise(transform, fixture.get_column(series), cuts=cuts or None)

    original = run_check(frame, report["cuts"])
    if original.get("cuts") != report["cuts"] or original.get("difference") != report["difference"]:
        raise ValueError("The original split did not reproduce the reported failure")
    reduced, check = _reduce(run_check, frame, original)

    with ExitStack() as cleanup:
        dest.mkdir()  # Only one writer can claim this directory.
        cleanup.callback(dest.rmdir)

        def create(name):
            path = dest / name
            file = path.open("xb")
            cleanup.callback(path.unlink, missing_ok=True)
            return file

        with create("fixture.parquet") as file:
            reduced.write_parquet(file)
        fixture = pl.read_parquet(dest / "fixture.parquet")
        try:
            assert_frame_equal(reduced, fixture, check_exact=True)
        except AssertionError as err:
            raise ValueError("Parquet round trip changed the reduced fixture") from err
        verified = run_check(fixture, check["cuts"])
        if verified != check:
            raise ValueError("The saved fixture did not reproduce the reduced failure")
        meta = dict(version=_FRAME_VERSION, transform=ref, original_input_rows=frame.height,
                    original_cuts=original["cuts"], input_rows=fixture.height,
                    cuts=verified["cuts"], difference=verified["difference"])
        if series is not None:
            meta.update(version=_SERIES_VERSION, mode="series", series=series)
        elif expr:
            meta.update(version=_EXPR_VERSION, mode="expr")
        if workdir is not None:
            meta["working_directory"] = _bundle_path(Path(workdir).resolve(), dest)
        # A short elementwise witness tests output length but has no interior split.
        short = (expr or series is not None) and fixture.height < 2
        expected = (2, "insufficient") if short else (0, "pass")
        # Pytest imports test files by basename in its default collection mode.
        with create(f"test_batchseam_{uuid4().hex}.py") as file:
            file.write(f"{_TEST}\n_EXPECTED = {expected!r}\n".encode("utf-8"))
        with create("repro.json") as file:
            file.write((json.dumps(meta, indent=2, allow_nan=False) + "\n").encode("utf-8"))
        cleanup.pop_all()
    return meta


def load_repro(dest) -> tuple[str, pl.DataFrame, list[int]]:
    """Load a DataFrame bundle using the legacy three-item return value."""
    spec, fixture, cuts, series = load_repro_config(dest)
    if series is not None:
        raise ValueError("Use load_repro_config for a Series bundle to retain the selected column.")
    return spec, fixture, cuts


def load_repro_config(dest) -> tuple[str, pl.DataFrame, list[int], str | None]:
    """Load a frame or Series bundle, resolving its source reference."""
    return _load_repro_case(dest, allow_expr=False)[:4]


def load_repro_case(dest) -> tuple[str, pl.DataFrame, list[int], str | None, bool]:
    """Return (spec, fixture, cuts, series, expr), resolving spec relative to the bundle."""
    return _load_repro_case(dest, allow_expr=True)[:5]


def _load_repro_case(dest, *, allow_expr):
    dest = Path(dest).resolve()
    manifest = dest / "repro.json"
    try:
        meta = json.loads(manifest.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        raise ValueError(f"Invalid repro manifest {manifest}: {err}") from err
    if (not isinstance(meta, dict) or type(meta.get("version")) is not int
            or meta["version"] not in (_FRAME_VERSION, _SERIES_VERSION, _EXPR_VERSION)):
        raise ValueError("Unsupported repro bundle version")
    expr = meta["version"] == _EXPR_VERSION
    series = None
    if expr:
        if not allow_expr:
            raise ValueError("Use load_repro_case for an expression bundle to retain its mode.")
        if meta.get("mode") != "expr" or "series" in meta:
            raise ValueError("Invalid expression mode or Series column in repro.json")
    elif meta["version"] == _SERIES_VERSION:
        series = meta.get("series")
        if meta.get("mode") != "series" or not isinstance(series, str):
            raise ValueError("Invalid Series mode or column in repro.json")
    elif "mode" in meta or "series" in meta:
        raise ValueError("Frame repro bundles cannot specify a mode or Series column")
    spec, cuts, rows = meta.get("transform"), meta.get("cuts"), meta.get("input_rows")
    if not isinstance(spec, str):
        raise ValueError("Invalid transform reference in repro.json")
    if spec not in (_EXPR_DEMOS if expr else _DEMOS):
        source, name = _file_spec(spec)
        spec = f"{(dest / source).resolve()}:{name}"
    fixture = pl.read_parquet(dest / "fixture.parquet")
    if type(rows) is not int or rows != fixture.height:
        raise ValueError("Repro input row count does not match fixture.parquet")
    if (not isinstance(cuts, list) or any(type(c) is not int or not 0 < c < rows for c in cuts)
            or any(a >= b for a, b in zip(cuts, cuts[1:]))):
        raise ValueError("Repro cuts must be strictly increasing interior offsets")
    whole_failure = False
    if series is not None and fixture.columns != [series]:
        raise ValueError("Repro Series column does not match fixture.parquet")
    if expr or series is not None:
        mode = "Expression" if expr else "Series"
        diff = meta.get("difference")
        if not isinstance(diff, dict) or diff.get("kind") not in ("length", "value", "schema", "batch_error"):
            raise ValueError(f"Invalid {mode} difference in repro.json")
        if diff["kind"] == "length":
            whole_failure = _validate_length(diff, cuts, rows, mode)
    if not cuts and not whole_failure:
        raise ValueError("Repro cuts must be strictly increasing interior offsets")
    workdir = meta.get("working_directory")
    if workdir is not None:
        if not isinstance(workdir, str) or not workdir:
            raise ValueError("Invalid working_directory in repro.json")
        workdir = (dest / workdir).resolve()
    return spec, fixture, cuts, series, expr, workdir


def _validate_length(diff, cuts, rows, mode):
    scope, expected, actual = diff.get("scope"), diff.get("expected"), diff.get("actual")
    if type(expected) is not int or type(actual) is not int or actual < 0 or actual == expected:
        raise ValueError(f"Invalid {mode} length difference in repro.json")
    if scope == "whole" and not cuts and "input_span" not in diff and expected == rows:
        return True
    if scope == "batch" and cuts:
        span = diff.get("input_span")
        for start, end in zip((0, *cuts), (*cuts, rows)):
            if span == [start, end] and all(type(offset) is int for offset in span) and expected == end - start:
                return False
    raise ValueError(f"{mode} length difference does not match the repro cuts and input rows")
