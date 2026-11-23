import polars as pl
import pytest

from batchseam import assert_elementwise, check_elementwise


def test_centering_changes_after_a_split():
    series = pl.Series("reading", [1, 2, 3, 4])

    def center(batch):
        return batch - batch.mean()

    assert check_elementwise(center, series) == dict(
        status="fail", input_rows=4, checked=1, cuts=[2],
        difference=dict(kind="value", output_row=0, column="reading", whole=-1.5, batched=-0.5))
    with pytest.raises(AssertionError, match=r"cuts \[2\].*output row 0"):
        assert_elementwise(center, series)


def test_filtering_can_match_across_batches_but_fails_whole_length():
    series = pl.Series("reading", [-1, 2, 0, 4])

    def positive(batch):
        return batch.filter(batch > 0)

    assert pl.concat([positive(series[:2]), positive(series[2:])]).equals(positive(series))
    assert check_elementwise(positive, series, cuts=[2]) == dict(
        status="fail", input_rows=4, checked=0, cuts=[],
        difference=dict(kind="length", scope="whole", expected=4, actual=2))
    with pytest.raises(AssertionError, match="whole series.*Expected 4 output rows, actual 2"):
        assert_elementwise(positive, series, cuts=[2])


def test_offsetting_batch_lengths_are_checked_before_concatenation():
    series = pl.Series("reading", [1, 2, 3, 4])

    def shift_boundary(batch):
        if len(batch) == 4:
            return batch
        if batch[0] == 1:
            return batch.head(1)
        return pl.concat([pl.Series(batch.name, [2]), batch])

    assert pl.concat([shift_boundary(series[:2]), shift_boundary(series[2:])]).equals(series)
    assert check_elementwise(shift_boundary, series, cuts=[2]) == dict(
        status="fail", input_rows=4, checked=1, cuts=[2],
        difference=dict(kind="length", scope="batch", input_span=[0, 2], expected=2, actual=1))
    with pytest.raises(AssertionError, match=r"cuts \[2\].*rows \[0, 2\).*Expected 2 output rows, actual 1"):
        assert_elementwise(shift_boundary, series, cuts=[2])


def test_later_batch_length_witness_uses_input_span():
    series = pl.Series([1, 2, 3, 4])

    def expand_last(batch):
        return pl.concat([batch, batch]) if batch[0] == 4 else batch

    assert check_elementwise(expand_last, series, cuts=[1, 3]) == dict(
        status="fail", input_rows=4, checked=1, cuts=[1, 3],
        difference=dict(kind="length", scope="batch", input_span=[3, 4], expected=1, actual=2))


def test_output_names_are_ignored_and_series_stay_unchanged():
    series = pl.Series("input", [1.0, None, float("nan"), -5.0])
    whole = (series * 2).cast(pl.Float32).alias("callback_full_output")

    def double(batch):
        assert batch.name == "input"
        return whole if len(batch) == 4 else (batch * 2).cast(pl.Float32).alias("callback_batch_output")

    assert check_elementwise(double, series, max_plans=2) == dict(status="pass", input_rows=4, checked=2,
        coverage={"column": "input", "plans": [[2], [1, 2]]})
    assert check_elementwise(double, series, exhaustive=True)["coverage"] == {"column": "input", "single_cut_range": [1, 3]}
    assert assert_elementwise(double, series, exhaustive=True) is None
    assert series.name == "input" and series.equals(pl.Series([1.0, None, float("nan"), -5.0]))
    assert whole.name == "callback_full_output"


def test_batch_dtype_must_match_whole_output_dtype():
    series = pl.Series("reading", [1, 2, 3])

    def cast_short(batch):
        return batch.cast(pl.Float32).alias("short") if len(batch) == 1 else batch.alias("long")

    assert check_elementwise(cast_short, series, cuts=[2])["difference"] == dict(
        kind="schema", input_span=[2, 3], whole=[["reading", "Int64"]], batched=[["reading", "Float32"]])


def test_values_are_compared_in_order():
    series = pl.Series("reading", [3, 4, 1, 2])
    assert check_elementwise(lambda batch: batch.sort(), series, cuts=[2])["difference"] == dict(
        kind="value", output_row=0, column="reading", whole=1, batched=3)


@pytest.mark.parametrize(("dtype", "delta"), [(pl.Float64, 1e-12), (pl.Datetime("ns"), 1)])
def test_native_values_are_compared_exactly(dtype, delta):
    series = pl.Series([1, 1])
    report = check_elementwise(lambda batch: (batch + len(batch) * delta).cast(dtype), series)
    assert report["difference"]["kind"] == "value"
    assert report["difference"]["output_row"] == 0


def test_default_planner_checks_null_boundaries():
    series = pl.Series([1] * 41 + [None] + [1] * 58)
    report = check_elementwise(lambda batch: batch.forward_fill(), series)
    assert report["cuts"] == [41]
    assert report["difference"] == dict(kind="value", output_row=41, column="", whole=1, batched=None)


@pytest.mark.parametrize("size", [0, 1])
def test_small_fixtures_still_check_whole_length(size):
    series = pl.Series(range(size), dtype=pl.Int64)
    assert check_elementwise(lambda batch: batch, series) == dict(status="insufficient", input_rows=size, checked=0)
    report = check_elementwise(lambda batch: pl.Series([1, 2]), series)
    assert report == dict(status="fail", input_rows=size, checked=0, cuts=[],
                          difference=dict(kind="length", scope="whole", expected=size, actual=2))
    with pytest.raises(AssertionError, match="At least two rows"):
        assert_elementwise(lambda batch: batch, series)


@pytest.mark.parametrize("options", [
    {"cuts": []}, {"cuts": [0]}, {"cuts": [3]}, {"cuts": [True]}, {"cuts": [2, 1]}, {"cuts": [1, 1]},
    {"max_plans": True}, {"max_plans": 0}, {"exhaustive": 1}, {"exhaustive": True, "cuts": [1]},
    {"exhaustive": True, "max_plans": 2},
])
def test_invalid_options_are_rejected_before_callback(options):
    def unexpected(batch):
        pytest.fail("Invalid options should stop the check before calling the callback")

    with pytest.raises(ValueError):
        check_elementwise(unexpected, pl.Series([1, 2, 3]), **options)


def test_wrong_return_types_follow_operational_error_conventions():
    series = pl.Series([1, 2, 3])
    with pytest.raises(ValueError, match="Whole-series callback failed.*must return a Polars Series"):
        assert_elementwise(lambda batch: batch.to_frame(), series)
    report = check_elementwise(lambda batch: batch if len(batch) > 1 else batch[0], series, cuts=[2])
    assert report["difference"] == dict(kind="batch_error", input_span=[2, 3],
                                        error="TypeError: callback must return a Polars Series")
