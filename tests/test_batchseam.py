import json
from pathlib import Path
import subprocess
import sys

import polars as pl
import pytest

from batchseam import check_batches


def check_cli(*args, cwd=None):
    return subprocess.run([sys.executable, "-I", "-m", "batchseam", *map(str, args)],
                          cwd=cwd, capture_output=True, text=True)


def test_demo_witness_replays_exactly(tmp_path):
    result = check_cli("demo", "--json", cwd=tmp_path)
    assert result.returncode == 1 and not result.stderr
    assert json.loads(result.stdout) == dict(status="fail", input_rows=4, checked=1, cuts=[2],
        difference=dict(kind="value", output_row=2, column="balance", whole=15, batched=4))
    assert check_cli("demo", "--cuts", "2", "--json", cwd=tmp_path).stdout == result.stdout


def test_equal_row_local_transform_with_special_values():
    df = pl.DataFrame({"x": [1.0, None, float("nan"), -5.0, 0.0], "a": [[1], [], None, [2, 3], []]})
    result = check_batches(lambda b: b.with_columns(y=pl.col("x") * 2), df)
    assert result == dict(status="pass", input_rows=5, checked=5)


def test_empty_intermediate_output_is_valid():
    df = pl.DataFrame({"x": [1, 2, 3, 4]})
    assert check_batches(lambda b: b.filter(pl.col("x") > 2), df)["status"] == "pass"


def test_binary_search_finds_earliest_difference_across_columns():
    df = pl.DataFrame({"x": [1, 0, 0, 2, 0]})
    def transform(b):
        return b.with_columns(a=pl.col("x") + b.height * (pl.col("x") == 2),
                              z=pl.col("x") + b.height * (pl.col("x") == 1))
    d = check_batches(transform, df, cuts=[3])["difference"]
    assert (d["output_row"], d["column"], d["whole"], d["batched"]) == (0, "z", 6, 4)


def test_float_comparison_is_exact():
    df = pl.DataFrame({"x": [1.0, 1.0]})
    result = check_batches(lambda b: b.with_columns(y=pl.col("x") + b.height * 1e-12), df)
    assert result["difference"]["column"] == "y"


def test_literal_column_name_that_looks_like_a_selector():
    df = pl.DataFrame({"x": [2, 3]})
    r = check_batches(lambda b: b.with_columns(pl.col("x").cum_sum().alias("*")), df)
    assert r["difference"] == dict(kind="value", output_row=1, column="*", whole=5, batched=3)


def test_nanosecond_difference_survives_even_when_python_scalar_would_round():
    df = pl.DataFrame({"x": [1, 1]})
    d = check_batches(lambda b: b.select(pl.lit(b.height).cast(pl.Datetime("ns")).alias("t")), df)["difference"]
    assert d["kind"] == "row_count"
    d = check_batches(lambda b: b.select((pl.col("x") + b.height).cast(pl.Datetime("ns")).alias("t")), df)["difference"]
    assert d == dict(kind="value", output_row=0, column="t")


def test_schema_and_partial_exception_are_witnesses():
    df = pl.DataFrame({"x": [1, 2, 3]})
    def schema(b):
        return b.cast(pl.Float64) if b.height == 1 else b
    d = check_batches(schema, df, cuts=[2])["difference"]
    assert d == dict(kind="schema", input_span=[2, 3], whole=[["x", "Int64"]], batched=[["x", "Float64"]])
    def reject_short(b):
        if b.height < 2:
            raise ValueError("too short")
        return b
    d = check_batches(reject_short, df, cuts=[2])["difference"]
    assert d == dict(kind="batch_error", input_span=[2, 3], error="ValueError: too short")


def test_multiple_cuts_and_small_budget():
    df = pl.DataFrame({"x": range(8)})
    assert check_batches(lambda b: b, df, max_plans=2)["checked"] == 2
    d = check_batches(lambda b: b.head(1), df, cuts=[1, 3, 7])["difference"]
    assert d == dict(kind="row_count", whole=1, batched=4)


@pytest.mark.parametrize("cuts", [[], [0], [3], [1, 1], [2, 1], [True]])
def test_bad_cuts(cuts):
    with pytest.raises(ValueError, match="interior offsets"):
        check_batches(lambda b: b, pl.DataFrame({"x": range(3)}), cuts=cuts)


@pytest.mark.parametrize("size", [0, 1])
def test_insufficient_fixture(size):
    assert check_batches(lambda b: b, pl.DataFrame({"x": range(size)})) == dict(
        status="insufficient", input_rows=size, checked=0)


def test_own_parquet_csv_and_invalid_transform(tmp_path):
    script = tmp_path / "my transforms.py"
    script.write_text("import helpers\ndef cents(b):\n print('checking')\n return helpers.convert(b)\n"
                      "def invalid(b): return None\n", encoding="utf-8")
    (tmp_path / "helpers.py").write_text("def convert(b): return b * 100\n", encoding="utf-8")
    df = pl.DataFrame({"amount": [4, 7, 9]})
    for ext in ["parquet", "csv"]:
        path = tmp_path / f"input.{ext}"
        getattr(df, f"write_{ext}")(path)
        r = check_cli("check", f"{script}:cents", path, "--json", cwd=Path(__file__).parent)
        assert r.returncode == 0 and json.loads(r.stdout)["status"] == "pass"
        assert "checking" in r.stderr
    r = check_cli("check", f"{script}:invalid", path, "--json")
    assert r.returncode == 2 and not r.stdout and "Whole-frame transform failed" in r.stderr
