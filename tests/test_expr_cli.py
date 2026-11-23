import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _run(cwd, *args, module="batchseam", pythonpath=None):
    flags, env = ["-I"], None
    if pythonpath is not None:
        flags, env = ["-P"], {**os.environ, "PYTHONPATH": str(pythonpath)}
    return subprocess.run([sys.executable, *flags, "-m", module, *map(str, args)],
                          cwd=cwd, env=env, capture_output=True, text=True, timeout=60)


def _report(result, code, status):
    assert result.returncode == code, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == status
    return report


@pytest.mark.parametrize(("target", "message"), [
    ("targets.py", "FILE.py:NAME"),
    ("targets.py:missing", "Polars Expr or zero-argument factory"),
    ("targets.py:value", "Polars Expr or zero-argument factory"),
    ("targets.py:wrong", "must return a Polars Expr"),
    ("targets.py:needs_arg", "failed when called with no arguments"),
])
def test_malformed_expression_targets(tmp_path, target, message):
    (tmp_path / "targets.py").write_text(
        "value = 5\ndef wrong(): return 5\ndef needs_arg(x): return x\n", encoding="utf-8",
    )
    result = _run(tmp_path, "check", target, EXAMPLES / "trades.csv", "--expr", "--json")
    assert result.returncode == 2 and not result.stdout
    assert message in result.stderr and "Traceback" not in result.stderr


def test_expr_and_series_are_mutually_exclusive(tmp_path):
    result = _run(tmp_path, "check", "missing.py:target", "input.csv", "--expr", "--series", "x")
    assert result.returncode == 2 and not result.stdout
    assert "not allowed with argument" in result.stderr


def test_fx_demo_and_file_target(tmp_path):
    bundle = tmp_path / "fx repro"
    result = _run(tmp_path, "demo", "fx", "--json", "--save-repro", bundle)
    assert not result.stderr
    demo = _report(result, 1, "fail")
    diff = demo["difference"]
    assert demo["cuts"] == [1]
    assert (diff["output_row"], diff["column"], diff["whole"], diff["batched"]) == (1, "usd", 10.0, 20.0)
    assert demo["repro"]["transform"] == "demo:fx" and demo["repro"]["mode"] == "expr"
    replay = _report(_run(tmp_path, "replay", bundle, "--json"), 1, "fail")
    assert replay["difference"] == diff
    human = _run(tmp_path, "demo", "fx")
    assert human.returncode == 1 and not human.stderr
    assert "Whole expression: 10.0" in human.stdout and "20.0" in human.stdout
    assert 'Input row values: {"amount": 10, "rate": 2.0}' in human.stdout

    fixture = EXAMPLES / "trades.csv"
    script = EXAMPLES / "fx.py"
    failed = _report(_run(tmp_path, "check", f"{script}:usd", fixture, "--expr", "--json"), 1, "fail")
    assert failed["difference"] == diff
    fixed = _run(tmp_path, "check", f"{script}:usd_fixed", fixture, "--expr", "--exhaustive")
    assert fixed.returncode == 0 and not fixed.stderr
    assert "Tested every single cut from 1 through 1." in fixed.stdout


@pytest.mark.parametrize(("body", "scope", "cuts", "expected", "actual", "label"), [
    ("pl.col('amount').head(1)", "whole", [], 2, 1, "Whole expression length mismatch"),
    ("pl.col('amount').map_batches(lambda s: s.head(0) if len(s) == 1 else s, return_dtype=pl.Int64)",
     "batch", [1], 1, 0, "Expression batch on input rows [0, 1) length mismatch"),
])
def test_expression_length_scopes(tmp_path, body, scope, cuts, expected, actual, label):
    (tmp_path / "targets.py").write_text(f"import polars as pl\nexpr = {body}\n", encoding="utf-8")
    args = ["check", "targets.py:expr", EXAMPLES / "trades.csv", "--expr", "--cuts", "1"]
    human = _run(tmp_path, *args)
    assert human.returncode == 1 and not human.stderr
    assert f"{label} (expected {expected}, actual {actual})" in human.stdout
    bundle = tmp_path / "repro"
    result = _report(_run(tmp_path, *args, "--json", "--save-repro", bundle), 1, "fail")
    diff = result["difference"]
    assert result["cuts"] == cuts and diff["kind"] == "length" and diff["scope"] == scope
    replay = _report(_run(tmp_path, "replay", bundle, "--json"), 1, "fail")
    assert replay["difference"] == diff and replay["cuts"] == cuts


def test_expr_factory_survives_configured_imports_in_generated_test(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    ops = src / "fx_ops.py"
    ops.write_text("import polars as pl\nexpr = pl.col('amount').cum_sum()\n", encoding="utf-8")
    (tmp_path / "job.py").write_text(
        "def build():\n    from fx_ops import expr\n    print('building expression')\n    return expr\n",
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text('[tool.pytest.ini_options]\npythonpath = ["src"]\n', encoding="utf-8")
    bundle = tmp_path / "repro"
    result = _run(tmp_path, "check", "job.py:build", EXAMPLES / "trades.csv", "--expr", "--json",
                  "--save-repro", bundle, pythonpath=src)
    assert result.stderr.strip() == "building expression"
    saved = _report(result, 1, "fail")["repro"]
    assert saved["mode"] == "expr" and saved["transform"] == "../job.py:build"
    red = _run(tmp_path, "-q", bundle, module="pytest")
    assert red.returncode == 1 and "batchseam replay exited with 1" in red.stdout
    ops.write_text("import polars as pl\nexpr = pl.col('amount') * 2\n", encoding="utf-8")
    green = _run(tmp_path, "-q", bundle, module="pytest")
    assert green.returncode == 0 and "1 passed" in green.stdout, green.stdout + green.stderr
