import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import batchseam
import polars


def test_suite_works_from_user_site(tmp_path):
    python = sys._base_executable
    env = {k:v for k,v in os.environ.items() if k.upper() not in {"PYTHONPATH", "PYTHONNOUSERSITE"}}
    env["PYTHONUSERBASE"] = str(tmp_path / "user install")

    def run(*args):
        return subprocess.run([python, "-P", *args], env=env, cwd=tmp_path, text=True,
                              capture_output=True, timeout=60)

    path = run("-c", "import site; print(site.getusersitepackages())")
    assert path.returncode == 0, path.stderr
    site = Path(path.stdout.strip())
    site.mkdir(parents=True)
    shutil.copytree(Path(batchseam.__file__).parent, site / "batchseam", ignore=shutil.ignore_patterns("__pycache__"))
    (site / "fixture_dependencies.pth").write_text(str(Path(polars.__file__).resolve().parents[1])+"\n",encoding="utf-8")
    (site / "user_only_helper.py").write_text("factor = 2\n")
    (tmp_path / "batchseam.py").write_text("raise RuntimeError('project file was imported before installed CLI')\n")
    (tmp_path / "calc.py").write_text("from user_only_helper import factor\nimport polars as pl\nexpr=pl.col('x')*factor\n")
    (tmp_path / "values.csv").write_text("x\n1\n2\n")
    (tmp_path / "batchseam.toml").write_text('version=1\n[[checks]]\nname="scale"\ntarget="calc.py:expr"\nfixture="values.csv"\nmode="expr"\n')
    origin=run("-c", "import batchseam;print(batchseam.__file__)")
    assert origin.returncode == 0 and Path(origin.stdout.strip()).is_relative_to(site)
    result=run("-m", "batchseam", "suite", "--json")
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["checks"][0]["status"] == "pass"

    # A CI job can supply its own project dependencies through a relative PYTHONPATH.
    libs = tmp_path / "lib"
    libs.mkdir()
    (libs / "job_helper.py").write_text("scale = 3\n")
    (tmp_path / "calc.py").write_text("from job_helper import scale\nimport polars as pl\nexpr=pl.col('x')*scale\n")
    env["PYTHONPATH"] = "lib"
    project = tmp_path / "rules"
    project.mkdir()
    (project / "batchseam.toml").write_text((tmp_path / "batchseam.toml").read_text().replace(
        "calc.py", "../calc.py").replace("values.csv", "../values.csv"))
    result = run("-m", "batchseam", "suite", "rules/batchseam.toml", "--json")
    assert result.returncode == 0, result.stdout + result.stderr
