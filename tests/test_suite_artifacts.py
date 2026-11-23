import json
from pathlib import Path
import runpy
import shutil

import pytest

from batchseam import suite
from batchseam.repro import load_repro_case


@pytest.fixture
def portfolio(tmp_path, monkeypatch):
    project = tmp_path / "portfolio project"
    shutil.copytree(Path(__file__).resolve().parents[1] / "examples", project)
    monkeypatch.chdir(tmp_path)
    return project / "batchseam.toml"


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_portfolio_saves_a_reduced_regression_that_turns_green(portfolio):
    config = portfolio.read_text(encoding="utf-8").replace("fx.py:usd_fixed", "fx.py:usd")
    portfolio.write_text(config, encoding="utf-8")
    trades = portfolio.with_name("trades.csv")
    trades.write_text(trades.read_text(encoding="utf-8").rstrip() + "\n20,3.0\n30,4.0\n", encoding="utf-8")
    output = Path("CI results")
    report = suite.run_suite(portfolio.relative_to(Path.cwd()), save_failures=output)
    failed, passed = report["checks"]
    assert (report["status"], failed["status"], passed["status"]) == ("fail", "fail", "pass")
    assert failed["artifact"] == "check-0001" and "artifact" not in passed
    assert "artifact" not in failed["report"]
    assert {p.name for p in output.iterdir()} == {"check-0001", "summary.json"}
    assert _json(output / "summary.json") == report

    bundle = output / failed["artifact"]
    meta = _json(bundle / "repro.json")
    assert not Path(meta["transform"].rpartition(":")[0]).is_absolute()
    spec, fixture, cuts, series, expr = load_repro_case(bundle)
    fx = portfolio.with_name("fx.py")
    assert spec == f"{fx}:usd" and series is None and expr
    assert 2 <= fixture.height < failed["report"]["input_rows"] and cuts == meta["cuts"]
    test, = bundle.glob("test_batchseam_*.py")
    regression = runpy.run_path(str(test))["test_batchseam_regression"]
    with pytest.raises(AssertionError, match="batchseam replay exited with 1"):
        regression()
    fx.write_text(fx.read_text(encoding="utf-8") + "\nusd = usd_fixed\n", encoding="utf-8")
    regression()


def test_all_pass_summary(portfolio):
    output = Path("all-pass")
    report = suite.run_suite(portfolio, save_failures=str(output))
    assert report == suite.run_suite(portfolio) and report["status"] == "pass"
    assert list(output.iterdir()) == [output / "summary.json"]
    assert _json(output / "summary.json") == report


def test_validates_every_check_before_creating_output(portfolio, monkeypatch):
    with portfolio.open("a", encoding="utf-8") as file:
        file.write('\n[[checks]]\nname = "bad config"\n')
    monkeypatch.setattr(suite.subprocess, "run", lambda *args, **kwargs: pytest.fail("check launched"))
    output = Path("invalid")
    with pytest.raises(ValueError, match="bad config.*missing keys"):
        suite.run_suite(portfolio, save_failures=output)
    assert not output.exists()


def test_existing_output_and_missing_parent_are_refused(portfolio, monkeypatch):
    output = Path("existing")
    output.mkdir()
    marker = output / "summary.json"
    marker.write_text("previous run", encoding="utf-8")
    monkeypatch.setattr(suite.subprocess, "run", lambda *args, **kwargs: pytest.fail("check launched"))
    with pytest.raises(FileExistsError):
        suite.run_suite(portfolio, save_failures=output)
    with pytest.raises(FileNotFoundError):
        suite.run_suite(portfolio, save_failures="absent/results")
    assert list(output.iterdir()) == [marker] and marker.read_text(encoding="utf-8") == "previous run"
    assert not Path("absent").exists()


def test_bundle_save_error_does_not_prevent_remaining_reports(portfolio, monkeypatch):
    config = portfolio.read_text(encoding="utf-8").replace("fx.py:usd_fixed", "fx.py:usd")
    config += '''
[[checks]]
name = "short fixture"
target = "transforms.py:cents"
fixture = "short.csv"
mode = "frame"

[[checks]]
name = "running total"
target = "transforms.py:running_total"
fixture = "amounts.csv"
mode = "frame"
'''
    portfolio.write_text(config, encoding="utf-8")
    portfolio.with_name("short.csv").write_text("amount\n1\n", encoding="utf-8")
    output = Path.cwd() / "mixed"
    blocked = output / "check-0001"
    run = suite.subprocess.run

    def block_first_bundle(args, **kwargs):
        if Path(args[args.index("--save-repro") + 1]) == blocked:
            blocked.write_text("already here", encoding="utf-8")
        return run(args, **kwargs)

    monkeypatch.setattr(suite.subprocess, "run", block_first_bundle)
    report = suite.run_suite(portfolio, save_failures=output)
    errored, passed, short, failed = report["checks"]
    assert report["status"] == "error"
    assert [item["status"] for item in report["checks"]] == ["error", "pass", "insufficient", "fail"]
    assert "Repro destination already exists" in errored["diagnostic"]
    assert all("artifact" not in item for item in (errored, passed, short))
    assert failed["artifact"] == "check-0004"
    assert {p.name for p in output.iterdir()} == {blocked.name, "check-0004", "summary.json"}
    assert blocked.read_text(encoding="utf-8") == "already here"
    assert _json(output / "summary.json") == report


def test_final_summary_is_only_visible_after_atomic_publish(portfolio, monkeypatch):
    output = Path.cwd() / "interrupted"
    summary = output / "summary.json"

    def interrupt_publish(staged, target):
        assert target == summary and not target.exists()
        assert _json(staged)["status"] == "pass"
        raise KeyboardInterrupt

    monkeypatch.setattr(Path, "replace", interrupt_publish)
    with pytest.raises(KeyboardInterrupt):
        suite.run_suite(portfolio, save_failures=output)
    assert output.is_dir() and not summary.exists()
