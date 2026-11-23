from collections.abc import Callable, Sequence

import polars as pl

from .core import check_batches


def assert_batch_invariant(transform: Callable[[pl.DataFrame], pl.DataFrame], frame: pl.DataFrame, *,
                           cuts: Sequence[int] | None = None, max_plans: int = 16,
                           exhaustive: bool = False) -> None:
    """Assert that every tested split matches the whole-frame result on this fixture."""
    report = check_batches(transform, frame, cuts=cuts, max_plans=max_plans, exhaustive=exhaustive)
    if report["status"] == "pass":
        return
    if report["status"] == "insufficient":
        raise AssertionError(f"At least two rows are needed to check batch invariance (got {report['input_rows']}).")

    diff = report["difference"]
    msg = f"Batch invariance failed at cuts {report['cuts']}"
    if diff["kind"] == "value":
        msg += f" at output row {diff['output_row']}, column {diff['column']!r}"
    elif diff["kind"] == "row_count":
        msg += " with different output row counts"
    else:
        start, end = diff["input_span"]
        msg += f" with {diff['kind']} on input rows [{start}, {end})"
    if "whole" in diff:
        msg += f". Expected {diff['whole']!r}, actual {diff['batched']!r}."
    if "error" in diff:
        msg += f". {diff['error']}"
    raise AssertionError(msg)
