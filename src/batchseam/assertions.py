from collections.abc import Callable, Sequence

import polars as pl

from .core import _DEFAULT_MAX_PLANS, check_batches, check_elementwise


def assert_batch_invariant(transform: Callable[[pl.DataFrame], pl.DataFrame], frame: pl.DataFrame, *,
                           cuts: Sequence[int] | None = None, max_plans: int = _DEFAULT_MAX_PLANS,
                           exhaustive: bool = False) -> None:
    """Assert that every tested split matches the whole-frame result on this fixture."""
    _assert_report(check_batches(transform, frame, cuts=cuts, max_plans=max_plans, exhaustive=exhaustive))


def assert_elementwise(callback: Callable[[pl.Series], pl.Series], series: pl.Series, *,
                       cuts: Sequence[int] | None = None, max_plans: int = _DEFAULT_MAX_PLANS,
                       exhaustive: bool = False) -> None:
    """Assert that each Series call preserves length and every tested split matches the whole result."""
    _assert_report(check_elementwise(callback, series, cuts=cuts, max_plans=max_plans, exhaustive=exhaustive),
                   elementwise=True)


def _assert_report(report, *, elementwise=False):
    if report["status"] == "pass":
        return
    if report["status"] == "insufficient":
        check = "elementwise behavior" if elementwise else "batch invariance"
        raise AssertionError(f"At least two rows are needed to check {check} (got {report['input_rows']}).")

    diff = report["difference"]
    kind = diff["kind"]
    check = "Elementwise check" if elementwise else "Batch invariance"
    msg = f"{check} failed"
    if diff.get("scope") == "whole":
        msg += " on the whole series"
    else:
        msg += f" at cuts {report['cuts']}"
    if kind == "value":
        msg += f" at output row {diff['output_row']}, column {diff['column']!r}"
    elif kind == "row_count":
        msg += " with different output row counts"
    elif "input_span" in diff:
        start, end = diff["input_span"]
        reason = "length mismatch" if kind == "length" else kind
        msg += f" with {reason} on input rows [{start}, {end})"
    if kind == "length":
        msg += f". Expected {diff['expected']} output rows, actual {diff['actual']}."
    if "whole" in diff:
        msg += f". Expected {diff['whole']!r}, actual {diff['batched']!r}."
    if "error" in diff:
        msg += f". {diff['error']}"
    raise AssertionError(msg)
