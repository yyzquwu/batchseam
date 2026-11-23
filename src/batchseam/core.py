"""Compare a whole-frame transform with the same transform on contiguous batches."""

from collections import deque
from collections.abc import Callable, Iterator, Sequence
from itertools import islice
from math import isfinite

import polars as pl
from polars.testing import assert_frame_equal


def _plans(rows: int) -> Iterator[tuple[int, ...]]:
    """Try a halfway split, a three-way split, the edges, then bisect gaps."""
    if rows < 2:
        return
    mid = rows // 2
    yield (mid,)
    if rows >= 3:
        yield (rows // 3, 2 * rows // 3)
    if mid != 1:
        yield (1,)
    if rows - 1 != mid:
        yield (rows - 1,)
    spans = deque([(2, mid - 1), (mid + 1, rows - 2)])
    while spans:
        lo, hi = spans.popleft()
        if lo <= hi:
            cut = (lo + hi) // 2
            yield (cut,)
            spans.extend([(lo, cut - 1), (cut + 1, hi)])


def _equal(left: pl.DataFrame, right: pl.DataFrame) -> bool:
    try:
        assert_frame_equal(left, right, check_exact=True)
    except AssertionError:
        return False
    return True


def _difference(whole, batched):
    if whole.height != batched.height:
        return dict(kind="row_count", whole=whole.height, batched=batched.height)
    if _equal(whole, batched):
        return None

    # Find the earliest differing output row using Polars' native precision.
    lo, hi = 0, whole.height - 1
    while lo < hi:
        mid = (lo + hi) // 2
        size = mid - lo + 1
        if _equal(whole.slice(lo, size), batched.slice(lo, size)):
            lo = mid + 1
        else:
            hi = mid
    for name in whole.columns:
        a = whole.get_column(name).slice(lo, 1).to_frame()
        b = batched.get_column(name).slice(lo, 1).to_frame()
        if not _equal(a, b):
            diff = dict(kind="value", output_row=lo, column=name)
            # Python conversion can discard nanoseconds on temporal values.
            # Show scalar values only when their JSON representation is lossless.
            dtype = a.dtypes[0]
            if dtype.is_integer() or dtype.is_float() or dtype in (pl.String, pl.Boolean):
                x, y = a.item(), b.item()
                if all(type(v) is not float or isfinite(v) for v in (x, y)):
                    diff.update(whole=x, batched=y)
            return diff
    raise RuntimeError("Could not locate the output difference")


def _schema(df):
    return [[name, str(dtype)] for name, dtype in df.schema.items()]


def check_batches(transform: Callable[[pl.DataFrame], pl.DataFrame], frame: pl.DataFrame, *,
                  cuts: Sequence[int] | None = None, max_plans: int = 16) -> dict:
    """Check a deterministic, stateless transform. Cut offsets and output rows are zero-based."""
    if type(max_plans) is not int or max_plans < 1:
        raise ValueError("max_plans must be a positive integer")
    rows = frame.height
    if cuts is None:
        plans = islice(_plans(rows), min(max_plans, rows))
    else:
        cuts = tuple(cuts)
        if (not cuts or any(type(c) is not int or not 0 < c < rows for c in cuts)
                or any(a >= b for a, b in zip(cuts, cuts[1:]))):
            raise ValueError(f"cuts must be strictly increasing interior offsets for {rows} rows")
        plans = [cuts]

    def apply(batch):
        out = transform(batch)
        if not isinstance(out, pl.DataFrame):
            raise TypeError("transform must return a Polars DataFrame")
        return out

    try:
        whole = apply(frame.clone())
    except Exception as err:
        raise ValueError(f"Whole-frame transform failed: {type(err).__name__}: {err}") from err
    report = dict(status="insufficient" if rows < 2 else "pass", input_rows=rows, checked=0)
    for plan in plans:
        parts = []
        report["checked"] += 1
        for start, end in zip((0, *plan), (*plan, rows)):
            try:
                part = apply(frame.slice(start, end - start))
            except Exception as err:
                diff = dict(kind="batch_error", input_span=[start, end], error=f"{type(err).__name__}: {err}")
                break
            if part.columns != whole.columns or part.dtypes != whole.dtypes:
                diff = dict(kind="schema", input_span=[start, end], whole=_schema(whole), batched=_schema(part))
                break
            parts.append(part)
        else:
            diff = _difference(whole, pl.concat(parts))
        if diff is not None:
            return dict(report, status="fail", cuts=list(plan), difference=diff)
    return report
