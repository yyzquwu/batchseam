import json
import subprocess
import sys

import polars as pl
import pytest

from batchseam import suite


def _check(name="example", target="transform.py:copy", fixture="input.csv", mode="frame", **options):
    fields = dict(name=name, target=target, fixture=fixture, mode=mode, **options)
    return "[[checks]]\n" + "".join(f"{key} = {json.dumps(value)}\n" for key, value in fields.items())


def _manifest(root, *checks):
    path = root / "checks.toml"
    path.write_text("version = 1\n" + "".join(checks), encoding="utf-8")
    return path


def test_suite_runs_all_modes_from_manifest_directory(tmp_path, monkeypatch):
    root = tmp_path / "project with spaces"
    root.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BATCHSEAM_SUITE_TEST", raising=False)
    script = root / "my transforms.py"
    script.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "import polars as pl\n"
        "from suite_helper import offset\n"
        "def running(batch):\n"
        "    assert Path('my fixture.csv').is_file()\n"
        "    os.environ['BATCHSEAM_SUITE_TEST'] = 'child only'\n"
        "    return batch.with_columns(pl.col('amount').cum_sum())\n"
        "def scale(batch):\n"
        "    assert 'BATCHSEAM_SUITE_TEST' not in os.environ\n"
        "    return batch + offset\n"
        "expr = pl.col('amount') * offset\n",
        encoding="utf-8",
    )
    (root / "suite_helper.py").write_text("offset = 2\n", encoding="utf-8")
    (root / "batchseam.py").write_text("raise AssertionError('shadowed the installed CLI')\n", encoding="utf-8")
    fixture = root / "my fixture.csv"
    pl.DataFrame({"amount": [1, 2, 3, 4], "--amount": [5, 6, 7, 8]}).write_csv(fixture)
    path = _manifest(
        root,
        _check("frame", f"{script.name}:running", fixture.name, cuts=[1, 3]),
        _check("series", f"{script}:scale", str(fixture), "series", column="--amount", max_plans=1),
        _check("expression", f"{script.name}:expr", fixture.name, "expr", exhaustive=True),
    )
    result = suite.run_suite(path.relative_to(tmp_path))
    assert (result["version"], result["status"]) == (1, "fail")
    frame, series, expr = result["checks"]
    assert [(row["name"], row["status"]) for row in (frame, series, expr)] == [
        ("frame", "fail"), ("series", "pass"), ("expression", "pass"),
    ]
    assert frame["report"]["cuts"] == [1, 3]
    assert series["report"]["coverage"] == dict(fixture=str(fixture), column="--amount", plans=[[2]])
    assert expr["report"]["coverage"]["single_cut_range"] == [1, 3]


def test_suite_distinguishes_runtime_errors_from_insufficient_and_continues(tmp_path):
    (tmp_path / "transform.py").write_text("def copy(batch): return batch\n", encoding="utf-8")
    (tmp_path / "input.csv").write_text("x\n1\n2\n", encoding="utf-8")
    (tmp_path / "short.csv").write_text("x\n1\n", encoding="utf-8")
    short = _check("short", fixture="short.csv", exhaustive=False)
    again = _check("again")
    assert suite.run_suite(_manifest(tmp_path, again))["status"] == "pass"
    assert suite.run_suite(_manifest(tmp_path, short, again))["status"] == "insufficient"

    result = suite.run_suite(_manifest(
        tmp_path, _check("missing", target="missing.py:copy"), short,
        _check("wrong symbol", target="transform.py:unknown"), again,
    ))
    assert result["status"] == "error"
    missing, insufficient, missing_symbol, passed = result["checks"]
    assert missing["name"] == "missing" and missing["status"] == "error"
    assert "missing.py" in missing["diagnostic"] and "report" not in missing
    assert insufficient == dict(name="short", status="insufficient",
                                report=dict(status="insufficient", input_rows=1, checked=0))
    assert missing_symbol["status"] == "error" and "Could not find callable" in missing_symbol["diagnostic"]
    assert passed["status"] == "pass"


@pytest.mark.parametrize(("invalid", "message"), [
    (_check("bad", extra=1), "check 'bad': unknown keys: extra"),
    (_check("bad").replace("fixture", "fixtur"), "check 'bad': unknown keys: fixtur"),
    (_check("bad").replace('fixture = "input.csv"\n', ""), "check 'bad': missing keys: fixture"),
    (_check(), "check 'example': duplicate name"),
    (_check(" "), "check #2: name"),
    (_check("bad", target="transform.py:"), "check 'bad': target"),
    (_check("bad", fixture=17), "check 'bad': fixture"),
    (_check("bad", mode="express"), "check 'bad': mode"),
    (_check("bad", mode="series"), "check 'bad': column is required"),
    (_check("bad", column="x"), "check 'bad': column is only allowed"),
    (_check("bad", cuts=[2, 1]), "check 'bad': cuts"),
    (_check("bad", cuts=[True]), "check 'bad': cuts"),
    (_check("bad", cuts="1,2"), "check 'bad': cuts"),
    (_check("bad", max_plans=True), "check 'bad': max_plans"),
    (_check("bad", exhaustive=1), "check 'bad': exhaustive"),
    (_check("bad", cuts=[1], max_plans=16), "check 'bad': .*mutually exclusive"),
    (_check("bad", cuts=[1], exhaustive=False), "check 'bad': .*mutually exclusive"),
    (_check("bad", max_plans=16, exhaustive=True), "check 'bad': .*mutually exclusive"),
])
def test_validates_every_check_before_launch(tmp_path, monkeypatch, invalid, message):
    monkeypatch.setattr(suite.subprocess, "run", lambda *args, **kwargs: pytest.fail("check was launched"))
    with pytest.raises(ValueError, match=message):
        suite.run_suite(_manifest(tmp_path, _check(), invalid))


@pytest.mark.parametrize(("contents", "message"), [
    ("", "version"),
    ("version = true", "version"),
    ("version = 2", "version"),
    ("version = 1", "at least one"),
    ("version = 1\nchecks = []", "at least one"),
    ("version = 1\nchecks = [1]", "check #1 must be a table"),
    ("version = 1\ncheck = []", "unknown keys: check"),
    ("version = 1\n[[checks]", "checks.toml"),
])
def test_invalid_whole_manifest(tmp_path, contents, message):
    path = tmp_path / "checks.toml"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        suite.run_suite(path)


def test_missing_manifest_raises_oserror(tmp_path):
    with pytest.raises(OSError):
        suite.run_suite(tmp_path / "missing.toml")


@pytest.mark.parametrize(("code", "stdout", "stderr", "diagnostic"), [
    (0, "garbage", "", "garbage"),
    (0, "[]", "", "code 0"),
    (0, '{"status":"pass"}', "", "code 0"),
    (1, '{"status":"pass","input_rows":2,"checked":1}', "", "code 1"),
    (2, "", "batchseam: invalid fixture", "invalid fixture"),
    (3, '{"status":"pass","input_rows":2,"checked":1}', "", "code 3"),
    (None, "", "", "cannot launch process"),
])
def test_suite_marks_bad_process_results_as_errors_and_keeps_running(
        tmp_path, monkeypatch, code, stdout, stderr, diagnostic):
    calls = []
    report = dict(status="pass", input_rows=2, checked=1)

    def run(args, *, cwd, **kwargs):
        assert args[:5] == [sys.executable, "-I", "-m", "batchseam", "check"]
        assert cwd == tmp_path
        calls.append(args)
        if len(calls) > 1:
            return subprocess.CompletedProcess(args, 0, json.dumps(report), "")
        if code is None:
            raise FileNotFoundError("cannot launch process")
        return subprocess.CompletedProcess(args, code, stdout, stderr)

    monkeypatch.setattr(suite.subprocess, "run", run)
    result = suite.run_suite(_manifest(tmp_path, _check("bad result"), _check("good result")))
    assert result["status"] == "error"
    bad, good = result["checks"]
    assert bad["status"] == "error" and diagnostic in bad["diagnostic"]
    assert good == dict(name="good result", status="pass", report=report)


def test_insufficient_takes_precedence_over_fail(tmp_path, monkeypatch):
    results = iter([
        (1, dict(status="fail", input_rows=2, checked=1, cuts=[1], difference={"kind": "value"})),
        (2, dict(status="insufficient", input_rows=1, checked=0)),
    ])

    def run(args, **kwargs):
        code, report = next(results)
        return subprocess.CompletedProcess(args, code, json.dumps(report), "")

    monkeypatch.setattr(suite.subprocess, "run", run)
    result = suite.run_suite(_manifest(tmp_path, _check("different"), _check("short")))
    assert result["status"] == "insufficient"
