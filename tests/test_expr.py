import json

import polars as pl
import polars.selectors as cs
import pytest

from batchseam import assert_expr_elementwise, check_batches, check_expr_elementwise


def test_inline_map_batches_lambda_is_evaluated_as_written():
    frame = pl.DataFrame({"amount": [1, 2, 3, 4]})
    expr = pl.col("amount").map_batches(lambda s: s - s.mean(), return_dtype=pl.Float64,
                                       is_elementwise=True).alias("centered")
    assert check_expr_elementwise(expr, frame) == dict(
        status="fail", input_rows=4, checked=1, cuts=[2],
        difference=dict(kind="value", output_row=0, column="centered", whole=-1.5, batched=-0.5,
                        input_values={"amount": 1}))
    with pytest.raises(AssertionError, match=r"cuts \[2\].*output row 0.*'centered'.*'amount': 1"):
        assert_expr_elementwise(expr, frame)


def test_configured_bound_method_passes():
    class Converter:
        def __init__(self, rate):
            self.rate = rate

        def convert(self, s):
            return s * self.rate

    frame = pl.DataFrame({"amount": [1, None, 3, 4]})
    expr = pl.col("amount").map_batches(Converter(120).convert, return_dtype=pl.Int64,
                                       is_elementwise=True).alias("converted")
    assert check_expr_elementwise(expr, frame, max_plans=2) == dict(
        status="pass", input_rows=4, checked=2, coverage={"column": "converted", "plans": [[2], [1, 2]]})
    assert assert_expr_elementwise(expr, frame) is None


def test_multi_column_map_batches_reports_values_from_the_first_differing_input_row():
    frame = pl.DataFrame({"amount": [10, 20, 30, 40], "rate": [2, 4, 5, 8]})
    expr = pl.map_batches(["amount", "rate"], lambda cols: cols[0] * cols[1][0],
                          return_dtype=pl.Int64, is_elementwise=True).alias("converted")
    assert check_expr_elementwise(expr, frame, cuts=[2]) == dict(
        status="fail", input_rows=4, checked=1, cuts=[2],
        difference=dict(kind="value", output_row=2, column="converted", whole=60, batched=150,
                        input_values={"amount": 30, "rate": 5}))


def test_filter_that_commutes_with_batching_still_fails_whole_length():
    frame = pl.DataFrame({"x": [-1, 2, 0, 4]})
    expr = pl.col("x").filter(pl.col("x") > 0)
    assert check_batches(lambda b: b.select(expr), frame, exhaustive=True)["status"] == "pass"
    assert check_expr_elementwise(expr, frame) == dict(
        status="fail", input_rows=4, checked=0, cuts=[],
        difference=dict(kind="length", scope="whole", expected=4, actual=2))
    with pytest.raises(AssertionError, match="whole frame.*Expected 4 output rows, actual 2"):
        assert_expr_elementwise(expr, frame)


def test_later_slice_must_preserve_length():
    frame = pl.DataFrame({"x": [1, 4, 3, 2]})
    expr = pl.col("x").filter(pl.col("x") >= pl.col("x").first())
    assert check_expr_elementwise(expr, frame, cuts=[2]) == dict(
        status="fail", input_rows=4, checked=1, cuts=[2],
        difference=dict(kind="length", scope="batch", expected=2, actual=1, input_span=[2, 4]))


def test_polars_return_dtype_error_on_a_slice_has_its_input_span():
    frame = pl.DataFrame({"x": [1, 2, 3]})
    expr = pl.col("x").map_batches(lambda s: s if len(s) > 1 else s.cast(pl.Float64), return_dtype=pl.Int64)
    diff = check_expr_elementwise(expr, frame, cuts=[2])["difference"]
    assert diff["kind"] == "batch_error"
    assert diff["input_span"] == [2, 3]
    assert "SchemaError" in diff["error"] and "Float64" in diff["error"]


def test_valid_single_column_selector_and_exhaustive_coverage():
    frame = pl.DataFrame({"x": [1, 2, 3, 4], "label": ["a", "b", "c", "d"]})
    assert check_expr_elementwise(cs.integer(), frame, exhaustive=True) == dict(
        status="pass", input_rows=4, checked=3, coverage={"column": "x", "single_cut_range": [1, 3]})
    assert check_expr_elementwise(pl.col("label"), frame, cuts=[1, 3])["coverage"]["plans"] == [[1, 3]]


@pytest.mark.parametrize(("expr", "match"), [
    (cs.boolean(), r"exactly one output column \(got 0\)"),
    (pl.all(), r"exactly one output column \(got 2\)"),
    (pl.col("missing"), "ColumnNotFoundError"),
])
def test_invalid_whole_selection_is_an_operational_error(expr, match):
    frame = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
    with pytest.raises(ValueError, match=f"Whole-frame expression failed.*{match}"):
        assert_expr_elementwise(expr, frame)


@pytest.mark.parametrize("size", [0, 1])
def test_small_fixtures_still_evaluate_the_whole_expression(size):
    frame = pl.DataFrame({"x": pl.Series(range(size), dtype=pl.Int64)})
    assert check_expr_elementwise(pl.col("x"), frame) == dict(status="insufficient", input_rows=size, checked=0)
    with pytest.raises(AssertionError, match="At least two rows"):
        assert_expr_elementwise(pl.col("x"), frame)
    assert check_expr_elementwise(pl.lit(pl.Series([1, 2])), frame)["difference"] == dict(
        kind="length", scope="whole", expected=size, actual=2)


def test_eager_select_does_not_broadcast_scalar_results():
    frame = pl.DataFrame({"x": [1, 2, 3]})
    assert check_expr_elementwise(pl.col("x").sum(), frame)["difference"] == dict(
        kind="length", scope="whole", expected=3, actual=1)


def test_input_witness_keeps_only_lossless_json_scalars_and_the_actual_output_name():
    frame = pl.DataFrame({
        "x": [8, 3, 4, -2],
        "text": ["a", "b", "c", "d"],
        "bool": [False, False, True, False],
        "ratio": [0.5, 0.5, 1.25, 0.5],
        "null": [None] * 4,
        "null_list": [[1], [], None, []],
        "list": [[1]] * 4,
        "binary": [b"bytes"] * 4,
        "time": pl.Series([1] * 4, dtype=pl.Datetime("ns")),
        "inf": [float("inf")] * 4,
        "nan": [float("nan")] * 4,
    })
    diff = check_expr_elementwise(pl.col("x").cum_sum().alias("*"), frame)["difference"]
    assert diff == dict(kind="value", output_row=2, column="*", whole=15, batched=4,
                        input_values={"x": 4, "text": "c", "bool": True, "ratio": 1.25,
                                      "null": None, "null_list": None})
    assert json.loads(json.dumps(diff, allow_nan=False)) == diff


@pytest.mark.parametrize("dtype", [pl.Float64, pl.Datetime("ns")])
def test_exact_ordered_comparison_preserves_native_precision(dtype):
    frame = pl.DataFrame({"x": [1, 1]})
    expr = (pl.col("x") + pl.len()).cast(dtype) if dtype == pl.Datetime("ns") else pl.col("x") + pl.len() * 1e-12
    diff = check_expr_elementwise(expr, frame)["difference"]
    assert diff["output_row"] == 0
    assert diff["input_values"] == {"x": 1}
    if dtype == pl.Datetime("ns"):
        assert "whole" not in diff and "batched" not in diff
    else:
        assert diff["whole"] != diff["batched"]


def test_api_requires_an_expression_and_an_eager_frame():
    frame = pl.DataFrame({"x": [1, 2]})
    with pytest.raises(TypeError, match="expr must be a Polars Expr"):
        check_expr_elementwise(lambda: pl.col("x"), frame)
    with pytest.raises(TypeError, match="frame must be a Polars DataFrame"):
        check_expr_elementwise(pl.col("x"), frame.lazy())


def test_invalid_plan_options_stop_before_whole_selection():
    frame = pl.DataFrame({"x": [1, 2]})
    with pytest.raises(ValueError, match="exhaustive cannot be combined"):
        check_expr_elementwise(pl.col("missing"), frame, cuts=[1], exhaustive=True)
