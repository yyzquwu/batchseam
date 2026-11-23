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


def test_nested_manifest_replay_and_short_series_witness_after_move(tmp_path):
    project = tmp_path / "original"
    rules = project / "rules"
    rules.mkdir(parents=True)
    (rules / "scale.txt").write_text("2")
    (rules / "data.csv").write_text("x\n1\n2\n")
    (rules / "one.csv").write_text("x\n-1\n")
    source = rules / "targets.py"
    source.write_text("from pathlib import Path\nimport polars as pl\n"
                      "k=int(Path('scale.txt').read_text())\n"
                      "expr = pl.col('x').cum_sum()*k\n"
                      "def positive(s): return s.filter(s>0)\n")
    manifest = ('version=1\n[[checks]]\nname="nested"\nmode="expr"\ntarget="targets.py:expr"\nfixture="data.csv"\n'
                '[[checks]]\nname="single"\nmode="series"\ntarget="targets.py:positive"\nfixture="one.csv"\ncolumn="x"\n')
    (rules / "batchseam.toml").write_text(manifest)
    result = invoke("suite", "rules/batchseam.toml", "--save-failures", "out", cwd=project)
    assert result.returncode == 1, result.stdout + result.stderr
    moved = project.rename(tmp_path / "moved")
    for ordinal in ("0001", "0002"):
        assert invoke("replay", f"out/check-{ordinal}", "--json", cwd=moved).returncode == 1
    source = moved / "rules/targets.py"
    source.write_text(source.read_text().replace(".cum_sum()", "").replace(".filter(s>0)", ""))
    short = invoke("replay", "out/check-0002", "--json", cwd=moved)
    assert short.returncode == 2 and json.loads(short.stdout)["status"] == "insufficient"
    tests = subprocess.run([sys.executable, "-I", "-m", "pytest", "-q", "out"], cwd=moved,
                           capture_output=True, timeout=90)
    assert tests.returncode == 0 and b"2 passed" in tests.stdout, tests.stdout + tests.stderr
