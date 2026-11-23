from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import runpy
from threading import Barrier

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from batchseam import repro
from batchseam.core import check_batches
from batchseam.repro import load_repro, save_repro


def _fill(frame):
    return frame.with_columns(pl.col("x").forward_fill())


def _total(frame):
    return frame.with_columns(pl.col("x").cum_sum())


def _save(transform, frame, dest, *, cuts=None, spec="demo:sensor"):
    return save_repro(transform, frame, check_batches(transform, frame, cuts=cuts), dest, spec=spec)


@pytest.mark.parametrize("values,cuts,expected", [
    ([None if i == 41 else i for i in range(100)], [41], [40, None]),
    ([0, 1, 2, None, 4, 5], [1, 3, 5], [2, None]),
])
def test_reduce_in_original_order_and_remap_cuts(tmp_path, values, cuts, expected):
    dest = tmp_path / "bundle"
    frame = pl.DataFrame({"x": values}, schema={"x": pl.Int16})
    meta = _save(_fill, frame, dest, cuts=cuts)
    spec, fixture, replay_cuts = load_repro(dest)
    assert spec == "demo:sensor"
    assert replay_cuts == meta["cuts"] == [1]
    assert meta["original_cuts"] == cuts
    assert meta["original_input_rows"] == len(values)
    assert meta["input_rows"] == 2
    assert_frame_equal(fixture, pl.DataFrame({"x": expected}, schema=frame.schema))
    assert check_batches(_fill, fixture, cuts=replay_cuts)["difference"] == meta["difference"]
    test, = dest.glob("test_batchseam_*.py")
    assert {p.name for p in dest.iterdir()} == {"fixture.parquet", "repro.json", test.name}


@pytest.mark.parametrize("kind", ["batch_error", "schema", "value"])
def test_reduction_rejects_unrelated_failures(tmp_path, kind):
    frame = pl.DataFrame({"x": [1, 2, 3, 4], "y": [1, 2, 3, 4]})

    def transform(batch):
        if batch.height == 1:
            if kind == "batch_error":
                raise ValueError("unrelated")
            if kind == "schema":
                return batch.cast(pl.Float64)
            return batch.with_columns(y=pl.lit(0, dtype=pl.Int64))
        if batch.height == 2 and batch[0, "x"] == 3:
            if kind == "batch_error":
                raise ValueError("original")
            if kind == "schema":
                return batch.cast(pl.Int32)
            return batch.with_columns(x=pl.lit(0, dtype=pl.Int64))
        return batch

    meta = _save(transform, frame, tmp_path / "bundle", cuts=[2])
    assert meta["input_rows"] == 4 and meta["cuts"] == [2]
    assert meta["difference"]["kind"] == kind


def test_whole_frame_errors_during_reduction_are_rejected(tmp_path):
    def transform(batch):
        if batch.height == 3 and batch[0, "x"] == 1:
            raise ValueError("unsupported size")
        return _total(batch)

    # The valid split must retain one row on each side even when larger candidates raise.
    meta = _save(transform, pl.DataFrame({"x": [1, 2, 3, 4, 5, 6]}), tmp_path / "bundle", cuts=[1])
    assert meta["input_rows"] == 2 and meta["cuts"] == [1]


@pytest.mark.parametrize("size", [0, 1, 4])
def test_only_failing_reports_create_bundles(tmp_path, size):
    frame = pl.DataFrame({"x": range(size)})
    with pytest.raises(ValueError, match="requires a failing report"):
        _save(lambda batch: batch, frame, tmp_path / "bundle")
    assert list(tmp_path.iterdir()) == []


def test_existing_targets_are_protected(tmp_path):
    frame = pl.DataFrame({"x": [1, 2]})
    directory, file = tmp_path / "empty", tmp_path / "file"
    directory.mkdir()
    file.write_bytes(b"keep me")
    for dest in (directory, file):
        with pytest.raises(FileExistsError, match="already exists"):
            _save(_total, frame, dest)
    assert list(directory.iterdir()) == []
    assert file.read_bytes() == b"keep me"


def test_concurrent_savers_cannot_overwrite(tmp_path, monkeypatch):
    frame, dest = pl.DataFrame({"x": [1, 2]}), tmp_path / "bundle"
    start = Barrier(2, timeout=10)

    def pause(transform, fixture, report):
        start.wait()
        return fixture, report

    monkeypatch.setattr(repro, "_reduce", pause)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [pool.submit(_save, _total, frame, dest) for _ in range(2)]
        errors = [result.exception(timeout=15) for result in results]
    assert errors.count(None) == 1
    assert isinstance(next(err for err in errors if err is not None), FileExistsError)
    _, saved, cuts = load_repro(dest)
    assert cuts == [1]
    assert_frame_equal(saved, frame)


def test_parquet_type_drift_rejects_an_otherwise_matching_failure(tmp_path, monkeypatch):
    def transform(batch):
        return _total(batch.cast(pl.Int64))

    frame = pl.DataFrame({"x": [1, 2]}, schema={"x": pl.Int16})
    read = pl.read_parquet
    monkeypatch.setattr(repro.pl, "read_parquet", lambda path: read(path).cast(pl.Int32))
    with pytest.raises(ValueError, match="Parquet round trip changed"):
        _save(transform, frame, tmp_path / "bundle")
    assert list(tmp_path.iterdir()) == []


def test_moved_bundle_regression_uses_current_source(tmp_path, monkeypatch):
    project = tmp_path / "old project"
    bundle = project / "tests" / "failure"
    bundle.parent.mkdir(parents=True)
    script = project / "my transforms.py"
    script.write_text("import polars as pl\ndef transform(b): return b.with_columns(pl.col('x').cum_sum())\n",
                      encoding="utf-8")
    frame = pl.DataFrame({"x": [1, 2]})
    meta = _save(_total, frame, bundle, spec=f"{script}:transform")
    assert not Path(meta["transform"].rpartition(":")[0]).is_absolute()
    assert json.loads((bundle / "repro.json").read_text(encoding="utf-8")) == meta

    moved = project.rename(tmp_path / "moved project")
    monkeypatch.chdir(tmp_path)
    script, bundle = moved / script.name, moved / bundle.relative_to(project)
    assert load_repro(bundle)[0] == f"{script.resolve()}:transform"
    test, = bundle.glob("test_batchseam_*.py")
    regression = runpy.run_path(str(test))["test_batchseam_regression"]
    with pytest.raises(AssertionError, match="batchseam replay exited with 1") as err:
        regression()
    assert '"status": "fail"' in str(err.value)
    script.write_text("def transform(b): return b\n", encoding="utf-8")
    regression()
