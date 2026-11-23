import json
from pathlib import Path
import subprocess
import sys

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def run(*args, cwd):
    return subprocess.run([sys.executable, "-I", "-m", "batchseam", "suite", *args], cwd=cwd,
                          capture_output=True, text=True, timeout=60)


def test_checked_in_suite(tmp_path):
    result = run(str(EXAMPLES / "batchseam.toml"), "--json", cwd=tmp_path)
    assert result.returncode == 0 and not result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "pass" and [c["name"] for c in report["checks"]] == ["trade conversion", "amount in cents"]


def test_default_manifest_human_output_and_global_errors(tmp_path):
    import shutil

    for name in ("fx.py", "trades.csv", "transforms.py", "amounts.csv"):
        shutil.copyfile(EXAMPLES / name, tmp_path / name)
    text = (EXAMPLES / "batchseam.toml").read_text().replace("fx.py:usd_fixed", "fx.py:usd")
    manifest = tmp_path / "batchseam.toml"
    manifest.write_text(text, encoding="utf-8")
    result = run(cwd=tmp_path)
    assert result.returncode == 1 and "trade conversion" in result.stdout and "amount in cents" in result.stdout
    manifest.write_text("version = 0", encoding="utf-8")
    result = run("--json", cwd=tmp_path)
    assert result.returncode == 2 and not result.stdout and "version" in result.stderr
