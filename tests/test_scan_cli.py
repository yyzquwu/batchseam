import json
import os
from pathlib import Path
import subprocess
import sys

import polars as pl


def _run(cwd, *args, module="batchseam"):
    return subprocess.run([sys.executable, "-I", "-m", module, *args], cwd=cwd,
                          capture_output=True, encoding="utf-8", timeout=60)


def _report(result, code):
    assert result.returncode == code, result.stdout + result.stderr
    assert not result.stderr
    return json.loads(result.stdout)


def test_named_template_creates_replayable_regression(tmp_path):
    project = tmp_path / "my project"
    project.mkdir()
    source = project / "day's transforms.py"
    script = (
        "from pathlib import Path\n"
        "import polars as pl\n"
        "Path(__file__).with_suffix('.loaded').touch()\n"
        "def center(series): return series - series.mean()\n"
        "def expression():\n"
        "    return pl.col('unrelated').map_batches(center, is_elementwise=True)\n"
    )
    source.write_text(script, encoding="utf-8")
    report = _report(_run(tmp_path, "scan", str(source), "--json"), 0)
    assert Path(report["root"]) == project and report["files_scanned"] == 1 and not report["errors"]
    finding, = report["findings"]
    assert finding["path"] == source.name and finding["line"] == 6
    assert finding["confidence"] == "polars" and finding["mode"] == "series"
    assert finding["argv"] == ["batchseam", "check", f"{source.name}:center", "FIXTURE.parquet", "--series", "COLUMN"]
    human = _run(tmp_path, "scan", str(source))
    assert human.returncode == 0 and not human.stderr
    assert str(project) in human.stdout and finding["command"] in human.stdout and finding["guidance"] in human.stdout
    assert not source.with_suffix(".loaded").exists()

    fixture, column = "actual input.parquet", "sales total"
    pl.DataFrame({column: [10, 30]}).write_parquet(project / fixture)
    replacements = {"FIXTURE.parquet": fixture, "COLUMN": column}
    argv = [replacements.get(arg, arg) for arg in finding["argv"]]
    checked = _report(_run(project, *argv[1:], "--json"), 1)
    assert checked["status"] == "fail" and checked["difference"]["column"] == column
    assert source.with_suffix(".loaded").exists()

    command = finding["command"].replace("FIXTURE.parquet", f"'{fixture}'").replace("COLUMN", f"'{column}'")
    command += " --save-repro regression --json"
    shell = (["powershell.exe", "-NoProfile", "-Command", command + "; exit $LASTEXITCODE"] if sys.platform == "win32"
             else ["sh", "-c", command])
    env = {**os.environ, "PATH": os.pathsep.join((str(Path(sys.executable).parent), os.environ.get("PATH", "")))}
    created = _report(subprocess.run(shell, cwd=project, env=env, capture_output=True, text=True, timeout=60), 1)
    assert created["repro"]["cuts"] == [1]
    regression = _run(project, "-q", "regression", module="pytest")
    assert regression.returncode == 1 and "batchseam replay exited with 1" in regression.stdout
    source.write_text(script.replace("series - series.mean()", "series * 2"), encoding="utf-8")
    regression = _run(project, "-q", "regression", module="pytest")
    assert regression.returncode == 0 and "1 passed" in regression.stdout


def test_manual_guidance_for_callback_shapes(tmp_path):
    (tmp_path / "callbacks.py").write_text(
        "import polars as pl\n"
        "def use(worker, other):\n"
        "    pl.col('x').map_batches(lambda s: s, is_elementwise=True)\n"
        "    pl.LazyFrame().map_batches(worker.convert, streamable=True)\n"
        "    pl.map_batches(['x', 'y'], combine, is_elementwise=True)\n"
        "    other.map_batches(make_callback(), is_elementwise=True)\n",
        encoding="utf-8",
    )
    report = _report(_run(tmp_path, "scan", "--json"), 0)
    first, bound, multi, ambiguous = report["findings"]
    assert "Extract the lambda into a named function" in first["guidance"]
    assert "assert_elementwise(callback, series)" in first["guidance"]
    assert "same configuration" in bound["guidance"]
    assert "assert_batch_invariant(instance.convert, frame)" in bound["guidance"]
    assert "every input at the same row offsets" in multi["guidance"]
    assert ambiguous["confidence"] == "ambiguous" and "Verify that this is a Polars call" in ambiguous["guidance"]
    human = _run(tmp_path, "scan")
    assert human.returncode == 0 and not human.stderr
    for finding in report["findings"]:
        assert finding["argv"] is None and finding["command"] is None and finding["target"] is None
        assert finding["guidance"] in human.stdout


def test_empty_partial_and_invalid_scans(tmp_path):
    empty = _report(_run(tmp_path, "scan", "--json"), 0)
    assert Path(empty["root"]) == tmp_path and not empty["findings"] and not empty["errors"]
    (tmp_path / "valid.py").write_text(
        "import polars as pl\npl.col('x').map_batches(lambda s: s, is_elementwise=True)\n", encoding="utf-8"
    )
    (tmp_path / "broken.py").write_text("def broken(\n", encoding="utf-8")
    partial = _report(_run(tmp_path, "scan", ".", "--json"), 2)
    assert len(partial["findings"]) == len(partial["errors"]) == 1
    human = _run(tmp_path, "scan", ".")
    assert human.returncode == 2 and not human.stderr
    assert "Scan incomplete" in human.stdout and "broken.py" in human.stdout and "valid.py" in human.stdout
    invalid = _report(_run(tmp_path, "scan", "absent.py", "--json"), 2)
    assert invalid["errors"] and not invalid["findings"] and invalid["files_scanned"] == 0


def test_named_frame_template_has_no_series_option(tmp_path):
    (tmp_path / "frames.py").write_text(
        "import polars as pl\n"
        "def identity(frame): return frame\n"
        "pl.LazyFrame().map_batches(identity, streamable=True)\n",
        encoding="utf-8",
    )
    finding, = _report(_run(tmp_path, "scan", "--json"), 0)["findings"]
    assert finding["argv"] == ["batchseam", "check", "frames.py:identity", "FIXTURE.parquet"]
    assert "COLUMN" not in finding["command"] and "COLUMN" not in finding["guidance"]
