import json
import subprocess
import sys

import polars as pl
import pytest


def _run(cwd, *args, module="batchseam"):
    return subprocess.run([sys.executable, "-I", "-m", module, *map(str, args)], cwd=cwd,
                          capture_output=True, text=True, timeout=60)


def _report(result, code, status):
    assert result.returncode == code, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == status
    return report


@pytest.mark.parametrize("column", ["", "*", "^sales.*$", "--amount"])
def test_check_exact_series_column(tmp_path, column):
    script = tmp_path / "transform.py"
    script.write_text(
        "import polars as pl\n"
        "def parse(batch):\n"
        f"    assert isinstance(batch, pl.Series) and batch.name == {column!r}\n"
        "    return batch.rename('date').str.to_date('%Y-%m-%d')\n",
        encoding="utf-8",
    )
    fixture = tmp_path / "dates.parquet"
    pl.DataFrame({"sales": [1, 2], column: [None, "2026-11-22"]}).write_parquet(fixture)
    args = ["check", "transform.py:parse", fixture, f"--series={column}", "--cuts", "1", "--json"]
    result = _run(tmp_path, *args, "--save-repro", "passed")
    assert not result.stderr and not (tmp_path / "passed").exists()
    assert _report(result, 0, "pass") == dict(status="pass", input_rows=2, checked=1,
        coverage={"fixture": str(fixture), "column": column, "plans": [[1]]})


def test_check_missing_series_column(tmp_path):
    (tmp_path / "transform.py").write_text("def transform(batch): return batch\n", encoding="utf-8")
    pl.DataFrame({"sales": [1, 2]}).write_csv(tmp_path / "input.csv")
    result = _run(tmp_path, "check", "transform.py:transform", "input.csv", "--series", "^sales$", "--json")
    assert result.returncode == 2 and not result.stdout
    assert "Series column '^sales$' was not found" in result.stderr


def test_plan_discovery_ignores_unselected_columns(tmp_path):
    (tmp_path / "transform.py").write_text(
        "def transform(batch): return batch + int(batch[0] == 41)\n", encoding="utf-8"
    )
    other = list(range(100))
    other[41] = None
    pl.DataFrame({"x": range(100), "other": other}).write_parquet(tmp_path / "input.parquet")
    args = ["check", "transform.py:transform", "input.parquet", "--series", "x", "--json"]
    assert _report(_run(tmp_path, *args, "--max-plans", "5"), 0, "pass")["checked"] == 5
    assert _report(_run(tmp_path, *args, "--cuts", "41"), 1, "fail")["cuts"] == [41]


@pytest.mark.parametrize(("rows", "status", "code"), [(0, "insufficient", 2), (1, "insufficient", 2), (4, "pass", 0)])
def test_check_series_budget_and_insufficient_input(tmp_path, rows, status, code):
    (tmp_path / "transform.py").write_text("def cents(batch): return batch * 100\n", encoding="utf-8")
    pl.DataFrame({"amount": range(rows)}, schema={"amount": pl.Int64}).write_parquet(tmp_path / "input.parquet")
    result = _run(tmp_path, "check", "transform.py:cents", "input.parquet", "--series", "amount",
                  "--max-plans", "1", "--json")
    assert not result.stderr
    expected = dict(status=status, input_rows=rows, checked=int(rows >= 2))
    if status == "pass":
        expected["coverage"] = {"fixture": "input.parquet", "column": "amount", "plans": [[2]]}
    assert _report(result, code, status) == expected


@pytest.mark.parametrize(("example", "code", "status"), [("dates", 0, "pass"), ("center", 1, "fail"),
                                                         ("positive", 1, "fail")])
def test_series_demos_and_replay(tmp_path, example, code, status):
    bundle = tmp_path / "demo repro"
    result = _run(tmp_path, "demo", example, "--save-repro", bundle, "--json")
    assert not result.stderr
    report = _report(result, code, status)
    if example == "dates":
        assert report["input_rows"] == 3 and report["checked"] > 0 and not bundle.exists()
        return
    if example == "center":
        assert report["cuts"] == [1]
        assert report["difference"] == dict(kind="value", output_row=0, column="amount", whole=-10.0, batched=0.0)
    else:
        assert report["checked"] == 0 and report["cuts"] == []
        assert report["difference"] == dict(kind="length", scope="whole", expected=3, actual=2)
        human = _run(tmp_path, "demo", example)
        assert human.returncode == 1 and not human.stderr
        assert "Whole Series length mismatch (expected 3, actual 2)" in human.stdout and "--cuts" not in human.stdout
    replay = _report(_run(tmp_path, "replay", bundle, "--json"), 1, "fail")
    assert replay["cuts"] == report["repro"]["cuts"]
    assert replay["difference"] == report["repro"]["difference"]


def test_check_batch_length_failure(tmp_path):
    (tmp_path / "transform.py").write_text(
        "def transform(batch): return batch if len(batch) > 1 else batch.head(0)\n", encoding="utf-8"
    )
    pl.DataFrame({"x": [10, 20, 30]}).write_csv(tmp_path / "input.csv")
    args = ["check", "transform.py:transform", "input.csv", "--series", "x", "--exhaustive"]
    report = _report(_run(tmp_path, *args, "--json"), 1, "fail")
    assert report["checked"] == 1 and report["cuts"] == [1]
    assert report["difference"] == dict(kind="length", scope="batch", expected=1, actual=0, input_span=[0, 1])
    result = _run(tmp_path, *args)
    assert result.returncode == 1 and not result.stderr
    assert "Batch on input rows [0, 1) length mismatch (expected 1, actual 0)" in result.stdout
    assert "Replay this split with --cuts 1" in result.stdout


@pytest.mark.parametrize(("expr", "kind"), [("batch - batch.mean()", "value"), ("batch.filter(batch > 0)", "length")])
def test_own_series_bundle_replays_current_callback(tmp_path, expr, kind):
    script = tmp_path / "my transforms.py"
    script.write_text(f"def transform(batch):\n    print('callback')\n    return {expr}\n", encoding="utf-8")
    fixture = tmp_path / "my input.parquet"
    pl.DataFrame({"": [-1, 2, 3], "unrelated": [None, "a", None]}).write_parquet(fixture)
    bundle = tmp_path / "repro"
    result = _run(tmp_path, "check", "my transforms.py:transform", fixture, "--series", "", "--json",
                  "--save-repro", bundle)
    report = _report(result, 1, "fail")
    assert "callback" in result.stderr
    assert report["difference"]["kind"] == kind
    assert pl.read_parquet(bundle / "fixture.parquet").columns == [""]
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    replay = _report(_run(elsewhere, "replay", bundle, "--json"), 1, "fail")
    assert replay["difference"] == report["repro"]["difference"]
    assert replay["cuts"] == report["repro"]["cuts"]
    regression = _run(elsewhere, "-q", bundle, module="pytest")
    assert regression.returncode == 1 and "batchseam replay exited with 1" in regression.stdout

    script.write_text("def transform(batch): return batch * 2\n", encoding="utf-8")
    _report(_run(elsewhere, "replay", bundle, "--json"), 0, "pass")
    regression = _run(elsewhere, "-q", bundle, module="pytest")
    assert regression.returncode == 0 and "1 passed" in regression.stdout


def test_cli_replays_legacy_dataframe_manifest(tmp_path):
    bundle = tmp_path / "legacy"
    bundle.mkdir()
    pl.DataFrame({"amount": [8, 3]}).write_parquet(bundle / "fixture.parquet")
    diff = dict(kind="value", output_row=1, column="balance", whole=11, batched=3)
    (bundle / "repro.json").write_text(json.dumps(dict(
        version=1, transform="demo:running-total", original_input_rows=2, original_cuts=[1],
        input_rows=2, cuts=[1], difference=diff,
    )), encoding="utf-8")
    result = _run(tmp_path, "replay", bundle, "--json")
    assert not result.stderr
    assert _report(result, 1, "fail") == dict(status="fail", input_rows=2, checked=1, cuts=[1], difference=diff)
