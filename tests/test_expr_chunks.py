import json
from pathlib import Path
import subprocess
import sys

import polars as pl

from batchseam import check_expr_elementwise


def test_chunked_input_matches_contiguous_fx_baseline(tmp_path):
    def group(rate):
        return pl.DataFrame({"amount": [10] * 100, "rate": [rate] * 100})

    frame = pl.concat([group(1.0), group(2.0)], rechunk=False)
    assert frame["rate"].n_chunks() == 2
    expr = pl.map_batches(["amount", "rate"], lambda xs: xs[0] * xs[1].first(),
                          return_dtype=pl.Float64, is_elementwise=True).alias("usd")
    expected = check_expr_elementwise(expr, frame.rechunk(), cuts=[100])
    assert expected["status"] == "fail"
    d = expected["difference"]
    assert (d["output_row"], d["whole"], d["batched"]) == (100, 10, 20)
    assert check_expr_elementwise(expr, frame, cuts=[100]) == expected

    script = Path(__file__).resolve().parents[1] / "examples/fx.py"
    for group_size in (100, 200):
        path = tmp_path / f"{group_size}.parquet"
        frame.write_parquet(path, row_group_size=group_size)
        run = subprocess.run([sys.executable, "-I", "-m", "batchseam", "check", f"{script}:usd",
                              str(path), "--expr", "--cuts", "100", "--json"],
                             capture_output=True, text=True, timeout=45)
        assert run.returncode == 1, run.stdout + run.stderr
        assert json.loads(run.stdout) == expected
