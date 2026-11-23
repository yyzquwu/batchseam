import json
import runpy
import subprocess
import sys

import polars as pl
import pytest

from batchseam import check_elementwise
from batchseam.repro import load_repro, load_repro_config, save_repro


def _replay(bundle, cwd, code):
    result = subprocess.run(
        [sys.executable, "-I", "-m", "batchseam", "replay", str(bundle), "--json"],
        cwd=cwd, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == code and not result.stderr, result.stdout + result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("expression,cuts,size,kind,scope", [
    ("s.filter(s > 0)", [2, 4], 2, "length", "whole"),
    ("(s - s.mean()).alias('centered')", [2, 4], 2, "value", None),
    ("s.head(1) if len(s) == 2 else s", [2], 3, "length", "batch"),
])
def test_series_bundle_replays_current_callback(tmp_path, expression, cuts, size, kind, scope):
    project = tmp_path / "old project"
    project.mkdir()
    script, bundle = project / "my callback.py", project / "repro"
    script.write_text(f"def callback(s):\n    return {expression}\n", encoding="utf-8")
    callback = runpy.run_path(str(script))["callback"]
    values = pl.Series("*", [1, 2, -3, 4, 5, 6], dtype=pl.Int16)
    frame = pl.DataFrame([values, pl.Series("unselected", ["private"] * len(values))])
    original = check_elementwise(callback, values, cuts=cuts)
    meta = save_repro(callback, frame, original, bundle, spec=f"{script}:callback", series=values.name)
    assert meta["mode"] == "series"
    assert meta["original_input_rows"] == len(values)
    assert meta["original_cuts"] == ([] if scope == "whole" else cuts)

    project = project.rename(tmp_path / "moved project")
    script, bundle = project / script.name, project / bundle.name
    spec, fixture, saved_cuts, series = load_repro_config(bundle)
    assert spec == f"{script.resolve()}:callback"
    assert series == values.name and fixture.columns == [series]
    assert fixture.dtypes == [pl.Int16] and fixture.height == size
    assert saved_cuts == meta["cuts"]
    replay = _replay(bundle, tmp_path, 1)
    diff = replay["difference"]
    assert replay["status"] == "fail" and diff == meta["difference"]
    assert diff["kind"] == kind and diff.get("scope") == scope
    if scope == "whole":
        assert saved_cuts == replay["cuts"] == [] and replay["checked"] == 0
        assert diff == dict(kind="length", scope="whole", expected=2, actual=1)
    elif scope == "batch":
        assert saved_cuts == [2]
        assert diff == dict(kind="length", scope="batch", expected=2, actual=1, input_span=[0, 2])
    else:
        assert diff["column"] == series

    test, = bundle.glob("test_batchseam_*.py")
    regression = runpy.run_path(str(test))["test_batchseam_regression"]
    with pytest.raises(AssertionError, match="batchseam replay exited with 1"):
        regression()
    script.write_text("def callback(s):\n    return s\n", encoding="utf-8")
    assert _replay(bundle, tmp_path, 0)["status"] == "pass"
    regression()


@pytest.mark.parametrize("rows", [0, 1])
def test_short_whole_length_bundle_replays_as_insufficient_after_fix(tmp_path, rows):
    script, bundle = tmp_path / "callback.py", tmp_path / "repro"
    script.write_text("def callback(s): return s.extend_constant(1, 1)\n", encoding="utf-8")
    callback = runpy.run_path(str(script))["callback"]
    values = pl.Series("", range(rows), dtype=pl.Int16)
    save_repro(callback, values.to_frame(), check_elementwise(callback, values), bundle,
               spec=f"{script}:callback", series="")
    assert load_repro_config(bundle)[3] == ""
    assert _replay(bundle, tmp_path, 1)["difference"] == dict(
        kind="length", scope="whole", expected=rows, actual=rows + 1,
    )
    script.write_text("def callback(s): return s\n", encoding="utf-8")
    assert _replay(bundle, tmp_path, 2) == dict(status="insufficient", input_rows=rows, checked=0)


def test_series_bundle_rejects_inconsistent_metadata(tmp_path):
    values = pl.Series("x", [-1, 2])

    def callback(s):
        return s.filter(s > 0)

    bundle = tmp_path / "repro"
    meta = save_repro(callback, values.to_frame(), check_elementwise(callback, values), bundle,
                      spec="demo:positive", series="x")
    with pytest.raises(ValueError, match="load_repro_config"):
        load_repro(bundle)
    spec, fixture, cuts, column = load_repro_config(bundle)
    assert spec == "demo:positive" and fixture.columns == [column] == ["x"] and cuts == []
    manifest = bundle / "repro.json"
    for fields, message in [
        (dict(version=1), "Frame repro bundles"),
        (dict(mode=None), "Series mode"),
        (dict(series=1), "Series mode or column"),
        (dict(series="missing"), "Series column does not match"),
        (dict(cuts=[True]), "interior offsets"),
        (dict(cuts=[1]), "length difference does not match"),
        (dict(difference=dict(meta["difference"], scope="batch")), "length difference does not match"),
        (dict(difference=dict(meta["difference"], expected=True)), "Invalid Series length difference"),
    ]:
        manifest.write_text(json.dumps(meta | fields), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_repro_config(bundle)

    manifest.write_text(json.dumps(meta), encoding="utf-8")
    fixture.with_columns(unselected=pl.lit(1)).write_parquet(bundle / "fixture.parquet")
    with pytest.raises(ValueError, match="Series column does not match"):
        load_repro_config(bundle)
