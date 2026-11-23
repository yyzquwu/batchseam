import json
from pathlib import Path
import os
import subprocess
import sys

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

def invoke(*args, cwd, env=None):
    return subprocess.run([sys.executable, "-I", "-m", "batchseam", *map(str,args)], cwd=cwd,
                          capture_output=True, env=env, timeout=75)

def test_version_matches_installed_distribution(tmp_path):
    from importlib.metadata import version
    p = invoke("--version", cwd=tmp_path)
    assert p.returncode == 0 and p.stdout.decode().strip() == f"batchseam {version('batchseam')}"

def test_ci_artifact_flow_and_utf8_markdown(tmp_path):
    import shutil
    for name in ("fx.py", "trades.csv", "transforms.py", "amounts.csv"):
        shutil.copyfile(EXAMPLES / name, tmp_path / name)
    config = (EXAMPLES/"batchseam.toml").read_text().replace("fx.py:usd_fixed","fx.py:usd").replace("trade conversion","東京")
    (tmp_path/"batchseam.toml").write_text(config, encoding="utf-8")
    env = {**os.environ, "PYTHONIOENCODING":"cp1252:strict"}
    p=invoke("suite","--markdown","--save-failures","out",cwd=tmp_path,env=env)
    assert p.returncode==1 and not p.stderr
    assert "東京" in p.stdout.decode("utf-8") and "input_values" not in p.stdout.decode("utf-8")
    result=json.loads((tmp_path/"out/summary.json").read_text(encoding="utf-8"))
    assert [c["status"] for c in result["checks"]]==["fail","pass"]
    bundle=tmp_path/"out/check-0001"
    assert invoke("replay",bundle,cwd=tmp_path).returncode==1
    with (tmp_path/"fx.py").open("a",encoding="utf-8") as f:
        f.write("\nusd = usd_fixed\n")
    assert invoke("replay",bundle,cwd=tmp_path).returncode==0
