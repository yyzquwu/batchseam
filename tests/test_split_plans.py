import polars as pl
import pytest

from batchseam import check_batches


@pytest.mark.parametrize("bad_starts", [(), (3, 7), (19, 22), (24,)])
def test_exhaustive_agrees_with_independent_single_cut_checks(bad_starts):
    frame = pl.DataFrame({"x": range(25)})

    def transform(batch):
        return batch.with_columns(pl.col("x") + (batch["x"][0] in bad_starts))

    whole = transform(frame)
    expected = next((cut for cut in range(1, frame.height)
                     if not whole.equals(pl.concat([transform(frame[:cut]), transform(frame[cut:])]))), None)

    report = check_batches(transform, frame, exhaustive=True)
    if expected is None:
        assert report == dict(status="pass", input_rows=25, checked=24)
    else:
        assert report["status"] == "fail"
        assert report["cuts"] == [expected]
        assert report["checked"] == expected
        replay = check_batches(transform, frame, cuts=[expected])
        assert replay == dict(report, checked=1)


@pytest.mark.parametrize("size", [0, 1])
def test_exhaustive_with_no_interior_cuts(size):
    frame = pl.DataFrame({"x": range(size)})
    assert check_batches(lambda batch: batch, frame, exhaustive=True) == dict(
        status="insufficient", input_rows=size, checked=0)


@pytest.mark.parametrize("options", [{"cuts": [1]}, {"max_plans": 2}])
def test_exhaustive_rejects_other_split_modes(options):
    with pytest.raises(ValueError, match="exhaustive"):
        check_batches(lambda batch: batch, pl.DataFrame({"x": range(3)}), exhaustive=True, **options)


@pytest.mark.parametrize(("edge", "cut"), [(0, 41), (-1, 42)])
def test_default_finds_both_boundaries_of_a_single_null(edge, cut):
    frame = pl.DataFrame({"x": [1] * 41 + [None] + [1] * 58})

    def transform(batch):
        return batch.drop_nulls() if batch["x"][edge] is None else batch

    report = check_batches(transform, frame)
    assert report["status"] == "fail"
    assert report["cuts"] == [cut]
    assert 1 < report["checked"] <= 16


def test_default_keeps_column_transitions_where_some_other_column_is_already_null():
    frame = pl.DataFrame({
        "always_null": [None] * 100,
        "text": ["a"] * 42 + [None] + ["a"] * 57,
        "*": [[1]] * 41 + [None] + [[1]] * 58,
    })

    def transform(batch):
        col = batch.get_column("*")
        return batch.filter(col.is_not_null()) if col[-1] is None else batch

    report = check_batches(transform, frame)
    assert report["status"] == "fail"
    assert report["cuts"] == [42]
    assert report["checked"] <= 16


def test_default_covers_null_transitions_across_scan_windows():
    offset = 4097
    frame = pl.DataFrame({"x": [1] * offset + [None] + [1] * 902})

    def transform(batch):
        return batch.drop_nulls() if batch["x"][0] is None else batch

    report = check_batches(transform, frame)
    assert report["cuts"] == [offset]
    assert report["checked"] <= 16


def test_default_is_deterministic_and_checks_distinct_plans_within_budget():
    frame = pl.DataFrame({"i": range(30), "x": [0, None] * 15})

    def run():
        plans = []

        def transform(batch):
            if batch["i"][0] == 0:
                plans.append([])
            if batch["i"][-1] < frame.height - 1:
                plans[-1].append(batch["i"][-1] + 1)
            return batch

        assert check_batches(transform, frame, max_plans=12)["checked"] == 12
        return [tuple(plan) for plan in plans[1:]]

    plans = run()
    assert plans == run()
    assert len(set(plans)) == len(plans) == 12
    assert {(15,), (10, 20), (1,), (29,)} <= set(plans)
