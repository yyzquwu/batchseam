"""Save a reduced, verified fixture for a failing batch split."""

from contextlib import ExitStack
import json
from os.path import relpath
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

from .core import check_batches

_DEMOS = {"demo:sensor", "demo:running-total"}
_MAX_ATTEMPTS = 128
_VERSION = 1
_TEST = '''from pathlib import Path
import subprocess
import sys


def test_batchseam_regression():
    result = subprocess.run(
        [sys.executable, "-I", "-m", "batchseam", "replay", str(Path(__file__).resolve().parent)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"batchseam replay exited with {result.returncode}\\n{result.stdout}{result.stderr}"
    )
'''


def _file_spec(spec):
    filename, sep, name = spec.rpartition(":")
    if not sep or not filename or not name.isidentifier():
        raise ValueError("Expected a transform reference of the form FILE.py:function")
    return Path(filename), name


def _saved_spec(spec, dest):
    if spec in _DEMOS:
        return spec
    source, name = _file_spec(spec)
    if not source.is_absolute():
        raise ValueError("Saving a repro requires an absolute transform path")
    try:
        source = Path(relpath(source, dest.resolve())).as_posix()
    except ValueError:  # Windows paths on different drives have no relative path.
        source = str(source)
    return f"{source}:{name}"


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
    return kind, diff["whole"], diff["batched"]


def _reduce(transform, frame, report):
    # Try progressively finer contiguous deletions within a fixed candidate budget.
    signature = _signature(report)
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
            if not cuts:
                continue
            span = list(map(remap, report["difference"].get("input_span", [])))
            candidate = pl.concat([frame.slice(0, start), frame.slice(end)])
            try:
                check = check_batches(transform, candidate, cuts=cuts)
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


def save_repro(transform, frame, report, dest, *, spec) -> dict:
    """Reduce a failing split and verify a new bundle against the saved Parquet file."""
    if report["status"] != "fail" or report["input_rows"] != frame.height:
        raise ValueError("Saving a repro requires a failing report for this fixture")
    dest = Path(dest).absolute()
    ref = _saved_spec(spec, dest)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError(f"Repro destination already exists: {dest}")
    original = check_batches(transform, frame, cuts=report["cuts"])
    if original.get("difference") != report["difference"]:
        raise ValueError("The original split did not reproduce the reported failure")
    reduced, check = _reduce(transform, frame, original)

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
        verified = check_batches(transform, fixture, cuts=check["cuts"])
        if verified != check:
            raise ValueError("The saved fixture did not reproduce the reduced failure")
        meta = dict(version=_VERSION, transform=ref, original_input_rows=frame.height,
                    original_cuts=original["cuts"], input_rows=fixture.height,
                    cuts=verified["cuts"], difference=verified["difference"])
        with create("test_regression.py") as file:
            file.write(_TEST.encode("utf-8"))
        with create("repro.json") as file:
            file.write((json.dumps(meta, indent=2, allow_nan=False) + "\n").encode("utf-8"))
        cleanup.pop_all()
    return meta


def load_repro(dest) -> tuple[str, pl.DataFrame, list[int]]:
    """Load the fixture and split, resolving user transforms from the bundle directory."""
    dest = Path(dest).resolve()
    meta = json.loads((dest / "repro.json").read_text(encoding="utf-8"))
    if not isinstance(meta, dict) or type(meta.get("version")) is not int or meta["version"] != _VERSION:
        raise ValueError("Unsupported repro bundle version")
    spec, cuts, rows = meta.get("transform"), meta.get("cuts"), meta.get("input_rows")
    if not isinstance(spec, str):
        raise ValueError("Invalid transform reference in repro.json")
    if spec not in _DEMOS:
        source, name = _file_spec(spec)
        spec = f"{(dest / source).resolve()}:{name}"
    fixture = pl.read_parquet(dest / "fixture.parquet")
    if type(rows) is not int or rows != fixture.height:
        raise ValueError("Repro input row count does not match fixture.parquet")
    if (not isinstance(cuts, list) or not cuts or any(type(c) is not int or not 0 < c < rows for c in cuts)
            or any(a >= b for a, b in zip(cuts, cuts[1:]))):
        raise ValueError("Repro cuts must be strictly increasing interior offsets")
    return spec, fixture, cuts
