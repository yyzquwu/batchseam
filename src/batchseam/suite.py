import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib


_REQUIRED = {"name", "target", "fixture", "mode"}
_SPLIT_OPTIONS = {"cuts", "max_plans", "exhaustive"}
_CHECK_KEYS = _REQUIRED | _SPLIT_OPTIONS | {"column"}
_EXIT_STATUS = {0: "pass", 1: "fail", 2: "insufficient"}


def _check_args(check, root):
    if unknown := check.keys() - _CHECK_KEYS:
        raise ValueError(f"unknown keys: {', '.join(sorted(unknown))}")
    if missing := _REQUIRED - check.keys():
        raise ValueError(f"missing keys: {', '.join(sorted(missing))}")
    name, target, fixture, mode = (check[key] for key in ("name", "target", "fixture", "mode"))
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a nonblank string")
    script, sep, symbol = target.rpartition(":") if isinstance(target, str) else ("", "", "")
    if not sep or not script.strip() or "\0" in script or not symbol.isidentifier():
        raise ValueError("target must have the form FILE.py:NAME")
    if not isinstance(fixture, str) or not fixture.strip() or "\0" in fixture:
        raise ValueError("fixture must be a nonblank file path")
    if mode not in ("expr", "frame", "series"):
        raise ValueError("mode must be 'expr', 'frame', or 'series'")

    args = [f"{root / script}:{symbol}", str(root / fixture), "--json"]
    if mode == "series":
        column = check.get("column")
        if not isinstance(column, str) or "\0" in column:
            raise ValueError("column is required for series mode and must be a string")
        args.append(f"--series={column}")
    elif "column" in check:
        raise ValueError("column is only allowed in series mode")
    elif mode == "expr":
        args.append("--expr")

    if len(check.keys() & _SPLIT_OPTIONS) > 1:
        raise ValueError("cuts, max_plans, and exhaustive are mutually exclusive")
    if "cuts" in check:
        cuts = check["cuts"]
        if (not isinstance(cuts, list) or not cuts or any(type(c) is not int or c < 1 for c in cuts)
                or any(a >= b for a, b in zip(cuts, cuts[1:]))):
            raise ValueError("cuts must be a nonempty array of strictly increasing positive integer offsets")
        args.extend(["--cuts", ",".join(map(str, cuts))])
    if "max_plans" in check:
        budget = check["max_plans"]
        if type(budget) is not int or budget < 1:
            raise ValueError("max_plans must be a positive integer")
        args.extend(["--max-plans", str(budget)])
    if "exhaustive" in check:
        if type(check["exhaustive"]) is not bool:
            raise ValueError("exhaustive must be a boolean")
        if check["exhaustive"]:
            args.append("--exhaustive")
    return args


def _read_suite(path):
    with path.open("rb") as file:
        manifest = tomllib.load(file)
    if unknown := manifest.keys() - {"version", "checks"}:
        raise ValueError(f"unknown keys: {', '.join(sorted(unknown))}")
    version = manifest.get("version")
    if type(version) is not int or version != 1:
        raise ValueError("version must be the integer 1")
    checks = manifest.get("checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("checks must contain at least one [[checks]] table")

    commands = {}
    for index, check in enumerate(checks, 1):
        if not isinstance(check, dict):
            raise ValueError(f"check #{index} must be a table")
        name = check.get("name")
        label = f"check {name!r}" if isinstance(name, str) and name.strip() else f"check #{index}"
        try:
            args = _check_args(check, path.parent)
            if name in commands:
                raise ValueError("duplicate name")
        except ValueError as err:
            raise ValueError(f"{label}: {err}") from err
        commands[name] = args
    return commands


def _run_check(root, name, args, *, artifact=None):
    if artifact is not None:
        args = [*args, "--save-repro", str(artifact)]
    try:
        # Start from the same installed package, including user-site installs, before any project code.
        env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
        result = subprocess.run([sys.executable, "-P", "-m", "batchseam", "check", *args],
                                cwd=root, env=env, capture_output=True, text=True, errors="replace")
    except OSError as err:
        return dict(name=name, status="error", diagnostic=f"Could not start batchseam check: {err}")

    expected = _EXIT_STATUS.get(result.returncode)
    try:
        report = json.loads(result.stdout)
    except (ValueError, RecursionError):
        report = None
    if (expected is not None and isinstance(report, dict) and report.get("status") == expected
            and all(type(report.get(key)) is int and report[key] >= 0 for key in ("input_rows", "checked"))
            and (expected != "fail" or (isinstance(report.get("cuts"), list)
                                        and isinstance(report.get("difference"), dict)))):
        item = dict(name=name, status=expected, report=report)
        if expected == "fail" and artifact is not None:
            if not isinstance(report.get("repro"), dict):
                return dict(name=name, status="error", diagnostic=f"batchseam check did not save a repro to {artifact}")
            item["artifact"] = artifact.name
        return item

    detail = result.stderr.strip() or result.stdout.strip() or "no output"
    return dict(name=name, status="error",
                diagnostic=f"batchseam check exited with code {result.returncode} without a valid report: {detail}")


def run_suite(path: str | Path, *, save_failures: str | Path | None = None) -> dict:
    """Validate a version 1 TOML manifest, then run its trusted checks in order.

    When save_failures is set, create a fresh directory under an existing parent.
    The saved files can contain original input values. A valid summary.json marks
    a completed run, including runs with check errors.
    """
    path = Path(path).absolute()
    try:
        commands = _read_suite(path)
    except ValueError as err:
        raise ValueError(f"{path}: {err}") from err
    output = None if save_failures is None else Path(save_failures).absolute()
    if output is not None:
        output.mkdir()
    checks = []
    for index, (name, args) in enumerate(commands.items(), 1):
        artifact = None if output is None else output / f"check-{index:04d}"
        checks.append(_run_check(path.parent, name, args, artifact=artifact))
    outcomes = {check["status"] for check in checks}
    status = next(status for status in ("error", "insufficient", "fail", "pass") if status in outcomes)
    report = dict(version=1, status=status, checks=checks)
    if output is not None:
        staged = output / ".summary.json.tmp"
        with staged.open("x", encoding="utf-8") as file:
            file.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        staged.replace(output / "summary.json")
    return report
