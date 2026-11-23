import json
import os
import subprocess
import sys

import polars as pl
import pytest

from batchseam import assert_batch_invariant


def _run(module, *args, cwd):
    return subprocess.run([sys.executable, "-I", "-m", module, *map(str, args)],
                          cwd=cwd, capture_output=True, text=True, timeout=60)


def test_failing_column_prints_through_legacy_pipe(tmp_path):
    name = "\u8ba1"
    script = tmp_path / "transform.py"
    script.write_text(f"def calc(b): return b.with_columns(b['x'].cum_sum().alias({name!r}))\n", encoding="utf-8")
    pl.DataFrame({"x": [2, 3]}).write_parquet(tmp_path / "x.parquet")
    result = subprocess.run([sys.executable, "-m", "batchseam", "check", "transform.py:calc", "x.parquet"],
                            cwd=tmp_path, env={**os.environ, "PYTHONIOENCODING": "cp1252:strict"},
                            capture_output=True, timeout=60)
    assert result.returncode == 1 and not result.stderr
    assert br"\u8ba1" in result.stdout and b"Whole frame: 5" in result.stdout


def test_sensor_demo_exhaustive_and_replay(tmp_path):
    result = _run("batchseam", "demo", "sensor", "--json", cwd=tmp_path)
    assert result.returncode == 1 and not result.stderr
    report = json.loads(result.stdout)
    assert report["input_rows"] == 100 and report["cuts"] == [41]
    assert (report["difference"]["whole"], report["difference"]["batched"]) == (40, None)

    bundle = tmp_path / "sensor repro"
    result = _run("batchseam", "demo", "sensor", "--exhaustive", "--save-repro", bundle, "--json", cwd=tmp_path)
    assert result.returncode == 1 and not result.stderr
    report = json.loads(result.stdout)
    assert report["checked"] == 41 and report["cuts"] == [41]
    assert report["repro"]["input_rows"] == 2 and report["repro"]["cuts"] == [1]
    result = _run("batchseam", "replay", bundle, "--json", cwd=tmp_path)
    assert result.returncode == 1 and not result.stderr
    assert json.loads(result.stdout) == dict(status="fail", input_rows=2, checked=1, cuts=[1],
        difference=dict(kind="value", output_row=1, column="reading", whole=40, batched=None))


@pytest.mark.parametrize("option", [("--max-plans", "16"), ("--cuts", "41")])
def test_exhaustive_rejects_budget_and_cuts(tmp_path, option):
    result = _run("batchseam", "demo", "sensor", "--exhaustive", *option, "--json", cwd=tmp_path)
    assert result.returncode == 2 and not result.stdout
    assert "not allowed with argument" in result.stderr


def test_running_total_bundle_human_output(tmp_path):
    bundle = tmp_path / "balance repro"
    result = _run("batchseam", "demo", "--save-repro", bundle, cwd=tmp_path)
    assert result.returncode == 1 and not result.stderr
    assert f"Saved repro to {bundle}" in result.stdout
    result = _run("batchseam", "replay", bundle, cwd=tmp_path)
    assert result.returncode == 1 and not result.stderr
    assert "2 input rows" in result.stdout and "column 'balance'" in result.stdout


def test_multiple_saved_regressions_collect_in_one_pytest_run(tmp_path):
    for name in ["sensor", "running-total"]:
        result = _run("batchseam", "demo", name, "--save-repro", tmp_path / name, cwd=tmp_path)
        assert result.returncode == 1 and not result.stderr
    result = _run("pytest", "-q", tmp_path, cwd=tmp_path)
    assert result.returncode == 1 and "2 failed" in result.stdout


def test_user_bundle_and_generated_test_track_current_source(tmp_path):
    source = tmp_path / "project files"
    source.mkdir()
    script = source / "my transforms.py"
    code = (
        "print('loading transform')\n"
        "def transform(batch):\n"
        "    from local_transform_support import pl\n"
        "    print('checking transform')\n"
        "    return batch.with_columns(balance=pl.col('amount').cum_sum())\n"
    )
    script.write_text(code, encoding="utf-8")
    (source / "local_transform_support.py").write_text("import polars as pl\n", encoding="utf-8")
    fixture = tmp_path / "my input.parquet"
    pl.DataFrame({"amount": [8, 3, 4, -2]}).write_parquet(fixture)
    bundle = tmp_path / "saved repro"
    alternate_cwd = tmp_path / "another directory"
    alternate_cwd.mkdir()

    result = _run("batchseam", "check", f"{script.relative_to(tmp_path)}:transform", fixture,
                  "--save-repro", bundle, "--json", cwd=tmp_path)
    assert result.returncode == 1 and json.loads(result.stdout)["status"] == "fail"
    assert "loading transform" in result.stderr and "checking transform" in result.stderr
    result = _run("batchseam", "replay", bundle, "--json", cwd=alternate_cwd)
    assert result.returncode == 1 and json.loads(result.stdout)["status"] == "fail"
    assert "checking transform" in result.stderr

    test, = bundle.glob("test_batchseam_*.py")
    result = _run("pytest", "-q", test, cwd=alternate_cwd)
    assert result.returncode == 1
    assert "batchseam replay exited with 1" in result.stdout and "column 'balance'" in result.stdout

    script.write_text(code.replace(".cum_sum()", " * 100"), encoding="utf-8")
    result = _run("pytest", "-q", test, cwd=alternate_cwd)
    assert result.returncode == 0 and "1 passed" in result.stdout
    result = _run("batchseam", "replay", bundle, "--json", cwd=alternate_cwd)
    assert result.returncode == 0 and json.loads(result.stdout)["status"] == "pass"


@pytest.mark.parametrize(("rows", "status", "exit_code"), [(2, "pass", 0), (1, "insufficient", 2)])
def test_save_skips_non_failing_reports(tmp_path, rows, status, exit_code):
    script = tmp_path / "transforms.py"
    script.write_text("def transform(batch): return batch\n", encoding="utf-8")
    fixture = tmp_path / "input.csv"
    pl.DataFrame({"x": range(rows)}).write_csv(fixture)
    bundle = tmp_path / "not created"
    result = _run("batchseam", "check", f"{script}:transform", fixture, "--save-repro", bundle, "--json", cwd=tmp_path)
    assert result.returncode == exit_code and not result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == status and "repro" not in report and not bundle.exists()


def test_replay_missing_bundle_is_a_cli_error(tmp_path):
    result = _run("batchseam", "replay", tmp_path / "missing", "--json", cwd=tmp_path)
    assert result.returncode == 2 and not result.stdout
    assert "batchseam:" in result.stderr


def test_assert_batch_invariant_preserves_input_errors_and_reports_witness():
    frame = pl.DataFrame({"x": [8, 3, 4]})

    def running_total(batch):
        return batch.with_columns(pl.col("x").cum_sum())

    assert assert_batch_invariant(lambda batch: batch, frame, exhaustive=True) is None
    with pytest.raises(AssertionError, match=r"cuts \[2\].*column 'x'.*Expected 15, actual 4"):
        assert_batch_invariant(running_total, frame, cuts=[2])
    with pytest.raises(AssertionError, match="At least two rows.*got 1"):
        assert_batch_invariant(running_total, frame.head(1))
    with pytest.raises(ValueError, match="interior offsets"):
        assert_batch_invariant(running_total, frame, cuts=[0])
    with pytest.raises(ValueError, match="Whole-frame transform failed.*must return a Polars DataFrame"):
        assert_batch_invariant(lambda batch: None, frame)
