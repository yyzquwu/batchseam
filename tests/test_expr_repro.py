import json
import os
from pathlib import Path
import py_compile
import runpy
import subprocess
import sys

import polars as pl
import pytest

from batchseam import repro
from batchseam.core import check_expr_elementwise
from batchseam.repro import load_repro, load_repro_case, load_repro_config, save_repro


def _cli(*args, cwd, code):
    result = subprocess.run(
        [sys.executable, "-I", "-m", "batchseam", *map(str, args), "--json"],
        cwd=cwd, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == code and not result.stderr, result.stdout + result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("body,size,kind,scope", [
    ("a.filter(a * b > 0)", 2, "length", "whole"),
    ("a.filter((pl.len() != 2) | (a * b > 0))", 3, "length", "batch"),
    ("a + b.cum_sum()", 2, "value", None),
    ("pl.map_batches([a, b], lambda cols: cols[0] * cols[1].cum_sum(), "
     "return_dtype=pl.Int64, is_elementwise=True)", 2, "value", None),
    ("pl.when(pl.len() == 2).then(pl.lit('bad')).otherwise((a + b).cast(pl.String)).cast(pl.Int64)",
     3, "batch_error", None),
])
def test_expression_bundle_replays_current_source(tmp_path, body, size, kind, scope):
    project = tmp_path / "original project"
    bundle = project / "tests" / "repro"
    bundle.parent.mkdir(parents=True)
    script = project / "my expressions.py"
    header = "import polars as pl\na, b = pl.col('a'), pl.col('b')\ndef build():\n    return "
    script.write_text(header + body + "\n", encoding="utf-8")
    expr = runpy.run_path(str(script))["build"]()
    frame = pl.DataFrame({"a": [10, 20, 30, 40, 50, 60], "b": [-1, 2, -3, 4, -5, 6]},
                         schema={"a": pl.Int16, "b": pl.Int16})
    report = check_expr_elementwise(expr, frame, cuts=[2, 4])
    meta = save_repro(expr, frame, report, bundle, spec=f"{script}:build", expr=True)
    assert (meta["version"], meta["mode"], meta["original_input_rows"]) == (3, "expr", 6)
    assert meta["original_cuts"] == ([] if scope == "whole" else [2, 4])
    assert "series" not in meta
    assert not Path(meta["transform"].rpartition(":")[0]).is_absolute()
    assert json.loads((bundle / "repro.json").read_text(encoding="utf-8")) == meta

    moved = project.rename(tmp_path / "moved project")
    script, bundle = moved / script.name, moved / bundle.relative_to(project)
    spec, fixture, cuts, series, is_expr = load_repro_case(bundle)
    assert spec == f"{script.resolve()}:build" and series is None and is_expr is True
    assert fixture.schema == frame.schema and fixture.height == size
    assert all(a == 10 * abs(b) for a, b in fixture.iter_rows())
    assert fixture["a"].is_sorted() and cuts == meta["cuts"]
    diff = meta["difference"]
    assert diff["kind"] == kind and diff.get("scope") == scope
    assert check_expr_elementwise(expr, fixture, cuts=cuts or None)["difference"] == diff
    assert _cli("replay", bundle, cwd=tmp_path, code=1)["difference"] == diff
    if scope == "whole":
        assert cuts == [] and diff == dict(kind="length", scope="whole", expected=2, actual=1)
    elif scope == "batch":
        assert cuts == [2]
        assert diff == dict(kind="length", scope="batch", expected=2, actual=1, input_span=[0, 2])
    elif kind == "value":
        assert diff["input_values"] == fixture.row(diff["output_row"], named=True)
    else:
        assert cuts == [2] and diff["input_span"] == [0, 2] and "bad" in diff["error"]

    test, = bundle.glob("test_batchseam_*.py")
    regression = runpy.run_path(str(test))["test_batchseam_regression"]
    with pytest.raises(AssertionError, match="batchseam replay exited with 1"):
        regression()
    script.write_text(header + "a + b\n", encoding="utf-8")
    assert _cli("replay", bundle, cwd=tmp_path, code=0)["status"] == "pass"
    regression()


def test_fx_demo_bundle(tmp_path):
    bundle = tmp_path / "fx"
    result = _cli("demo", "fx", "--save-repro", bundle, cwd=tmp_path, code=1)
    spec, fixture, cuts, series, expr = load_repro_case(bundle)
    assert spec == "demo:fx" and fixture.width >= 2 and series is None and expr is True
    replay = _cli("replay", bundle, cwd=tmp_path, code=1)
    assert replay["cuts"] == cuts == result["repro"]["cuts"]
    assert replay["difference"] == result["repro"]["difference"]


@pytest.mark.parametrize("rows", [0, 1])
def test_short_expression_bundle_regression_checks_whole_length(tmp_path, rows):
    script, bundle = tmp_path / "source.py", tmp_path / "repro"
    header = "import polars as pl\nexpr = pl.col('x')"
    script.write_text(header + ".extend_constant(1, 1)\n", encoding="utf-8")
    expr = runpy.run_path(str(script))["expr"]
    frame = pl.DataFrame({"x": range(rows)}, schema={"x": pl.Int16})
    meta = save_repro(expr, frame, check_expr_elementwise(expr, frame), bundle, spec=f"{script}:expr", expr=True)
    assert not meta["cuts"] and meta["difference"] == dict(
        kind="length", scope="whole", expected=rows, actual=rows + 1,
    )
    test, = bundle.glob("test_batchseam_*.py")
    regression = runpy.run_path(str(test))["test_batchseam_regression"]
    with pytest.raises(AssertionError, match="batchseam replay exited with 1"):
        regression()
    script.write_text(header + "\n", encoding="utf-8")
    regression()


def test_expression_bundle_validates_mode_source_and_length(tmp_path):
    frame = pl.DataFrame({"x": [-1, 2]})
    expr = pl.col("x").filter(pl.col("x") > 0)
    report = check_expr_elementwise(expr, frame)
    bundle = tmp_path / "repro"
    script = tmp_path / "source.py"
    script.write_text("import polars as pl\nexpr = pl.col('x').filter(pl.col('x') > 0)\n", encoding="utf-8")
    spec = f"{script}:expr"
    with pytest.raises(ValueError, match="existing source file"):
        save_repro(expr, frame, report, bundle, spec=f"{tmp_path / 'missing.py'}:build", expr=True)
    with pytest.raises(ValueError, match="cannot select a Series"):
        save_repro(expr, frame, report, bundle, spec=spec, series="x", expr=True)
    with pytest.raises(TypeError, match="requires a Polars Expr"):
        save_repro(lambda: expr, frame, report, bundle, spec=spec, expr=True)
    assert not bundle.exists()

    meta = save_repro(expr, frame, report, bundle, spec=spec, expr=True)
    for legacy in (load_repro, load_repro_config):
        with pytest.raises(ValueError, match="load_repro_case"):
            legacy(bundle)
    for fields, message in [
        (dict(mode="series"), "Invalid expression mode"),
        (dict(series=None), "Invalid expression mode"),
        (dict(cuts=[True]), "interior offsets"),
        (dict(cuts=[1]), "length difference does not match"),
        (dict(difference=dict(meta["difference"], expected=True)), "Invalid Expression length"),
    ]:
        (bundle / "repro.json").write_text(json.dumps(meta | fields), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_repro_case(bundle)


def test_generated_expression_test_uses_edited_packaged_source(tmp_path):
    package, bundle = tmp_path / "ops", tmp_path / "repro"
    package.mkdir()
    (package / "__init__.py").touch()
    script = package / "fx.py"
    header = "import polars as pl\nexpr = pl.col('x')"
    script.write_text(header + ".cum_sum()\n", encoding="utf-8")
    original = script.stat()
    py_compile.compile(str(script), doraise=True)
    expr = runpy.run_path(str(script))["expr"]
    frame = pl.DataFrame({"x": [1, 2]})
    save_repro(expr, frame, check_expr_elementwise(expr, frame), bundle, spec=f"{script}:expr", expr=True)
    test, = bundle.glob("test_batchseam_*.py")
    regression = runpy.run_path(str(test))["test_batchseam_regression"]
    with pytest.raises(AssertionError, match="batchseam replay exited with 1"):
        regression()

    script.write_text(header + " " * len(".cum_sum()") + "\n", encoding="utf-8")
    os.utime(script, ns=(original.st_atime_ns, original.st_mtime_ns))
    assert _cli("replay", bundle, cwd=tmp_path, code=0)["status"] == "pass"
    regression()


def test_expression_readback_rejects_a_changed_difference(tmp_path, monkeypatch):
    frame = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
    script, bundle = tmp_path / "source.py", tmp_path / "repro"
    script.write_text("import polars as pl\nexpr = pl.col('x') + pl.col('y').cum_sum()\n", encoding="utf-8")
    expr = runpy.run_path(str(script))["expr"]
    read = pl.read_parquet

    def read_parquet(path):
        # Keep the fixture intact and simulate drift in the expression's reported values.
        def changed(expr, fixture, **options):
            return check_expr_elementwise(expr + 1, fixture, **options)

        monkeypatch.setattr(repro.core, "check_expr_elementwise", changed)
        return read(path)

    monkeypatch.setattr(repro.pl, "read_parquet", read_parquet)
    with pytest.raises(ValueError, match="did not reproduce the reduced failure"):
        save_repro(expr, frame, check_expr_elementwise(expr, frame), bundle, spec=f"{script}:expr", expr=True)
    assert not bundle.exists()
