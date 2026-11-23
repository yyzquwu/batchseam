"""Compare a whole-input transform with the same transform on contiguous batches."""

from collections import deque
from collections.abc import Callable, Iterator, Sequence
from itertools import islice, zip_longest
from math import isfinite

import polars as pl
from polars.testing import assert_frame_equal


_DEFAULT_MAX_PLANS = 16
_NULL_SCAN_ROWS = 4096


def _ordinary_plans(rows: int) -> Iterator[tuple[int, ...]]:
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


def _null_cuts(frame: pl.DataFrame) -> Iterator[int]:
    cols = [col for col in frame.iter_columns() if 0 < col.null_count() < frame.height]
    if not cols:
        return
    # Include the previous row so each window also checks its left boundary.
    for start in range(1, frame.height, _NULL_SCAN_ROWS):
        size = min(_NULL_SCAN_ROWS, frame.height - start)
        changes = pl.repeat(False, size, eager=True)
        for col in cols:
            mask = col.slice(start - 1, size + 1).is_null()
            changes |= mask.head(size) != mask.tail(size)
        for offset in changes.arg_true():
            yield start + offset


def _plans(frame: pl.DataFrame) -> Iterator[tuple[int, ...]]:
    ordinary = _ordinary_plans(frame.height)
    seen = set()
    # Keep the midpoint, three-way split, and edges first when available.
    for plan in islice(ordinary, 4):
        seen.add(plan)
        yield plan
    nulls = ((cut,) for cut in _null_cuts(frame))
    for pair in zip_longest(nulls, ordinary):
        for plan in pair:
            if plan is not None and plan not in seen:
                seen.add(plan)
                yield plan


def _equal(left: pl.DataFrame, right: pl.DataFrame) -> bool:
    try:
        assert_frame_equal(left, right, check_exact=True)
    except AssertionError:
        return False
    return True


def _difference(whole, batched):
    if whole.height != batched.height:
        return dict(kind="row_count", whole=whole.height, batched=batched.height)
    if pl.Object in whole.dtypes:
        raise ValueError("Cannot reliably compare Python values in Polars Object output. "
                         "Return a native Polars dtype to check ordered values.")
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


def _input_values(frame, row):
    values = {}
    for col in frame.slice(row, 1).iter_columns():
        dtype = col.dtype
        if col.null_count():
            values[col.name] = None
        elif dtype.is_integer() or dtype.is_float() or dtype in (pl.String, pl.Boolean, pl.Categorical, pl.Enum, pl.Object):
            value = col.item()
            if type(value) in (str, bool, int) or (type(value) is float and isfinite(value)):
                values[col.name] = value
    return values


def _check(transform, data, cuts, max_plans, exhaustive, *, mode="frame"):
    if type(max_plans) is not int or max_plans < 1:
        raise ValueError("max_plans must be a positive integer")
    if type(exhaustive) is not bool:
        raise ValueError("exhaustive must be a boolean")
    series = mode == "series"
    elementwise = mode != "frame"
    frame = data.to_frame() if series else data
    rows = frame.height
    if exhaustive:
        if cuts is not None or max_plans != _DEFAULT_MAX_PLANS:
            raise ValueError("exhaustive cannot be combined with cuts or a nondefault max_plans")
        plans = ((cut,) for cut in range(1, rows))
    elif cuts is None:
        plans = islice(_plans(frame), min(max_plans, rows))
    else:
        cuts = tuple(cuts)
        if (not cuts or any(type(c) is not int or not 0 < c < rows for c in cuts)
                or any(a >= b for a, b in zip(cuts, cuts[1:]))):
            raise ValueError(f"cuts must be strictly increasing interior offsets for {rows} rows")
        plans = [cuts]

    out_type = pl.Series if series else pl.DataFrame
    call_name = "callback" if series else "transform"
    errors = (Exception, pl.exceptions.PanicException) if mode == "expr" else (Exception,)

    def apply(batch):
        out = transform(batch)
        if not isinstance(out, out_type):
            raise TypeError(f"{call_name} must return a Polars {out_type.__name__}")
        # Use the input name for diagnostics and preserve the user's output object.
        return out.to_frame(data.name) if series else out

    try:
        whole = apply(data.clone())
    except errors as err:
        call = {"frame": "Whole-frame transform", "series": "Whole-series callback",
                "expr": "Whole-frame expression"}[mode]
        raise ValueError(f"{call} failed: {type(err).__name__}: {err}") from err
    report = dict(status="insufficient" if rows < 2 else "pass", input_rows=rows, checked=0)
    if elementwise and whole.height != rows:
        diff = dict(kind="length", scope="whole", expected=rows, actual=whole.height)
        return dict(report, status="fail", cuts=[], difference=diff)
    tested = [] if elementwise and not exhaustive else None
    for plan in plans:
        if tested is not None:
            tested.append(list(plan))
        parts = []
        report["checked"] += 1
        for start, end in zip((0, *plan), (*plan, rows)):
            try:
                part = apply(data.slice(start, end - start))
            except errors as err:
                diff = dict(kind="batch_error", input_span=[start, end], error=f"{type(err).__name__}: {err}")
                break
            if elementwise and part.height != end - start:
                diff = dict(kind="length", scope="batch", expected=end - start, actual=part.height,
                            input_span=[start, end])
                break
            if part.columns != whole.columns or part.dtypes != whole.dtypes:
                diff = dict(kind="schema", input_span=[start, end], whole=_schema(whole), batched=_schema(part))
                break
            parts.append(part)
        else:
            diff = _difference(whole, pl.concat(parts))
        if diff is not None:
            if mode == "expr" and diff["kind"] == "value":
                diff["input_values"] = _input_values(frame, diff["output_row"])
            return dict(report, status="fail", cuts=list(plan), difference=diff)
    if elementwise and report["status"] == "pass":
        report["coverage"] = {"column": whole.columns[0]}
        if exhaustive:
            report["coverage"]["single_cut_range"] = [1, rows - 1]
        else:
            report["coverage"]["plans"] = tested
    return report


def check_batches(transform: Callable[[pl.DataFrame], pl.DataFrame], frame: pl.DataFrame, *,
                  cuts: Sequence[int] | None = None, max_plans: int = _DEFAULT_MAX_PLANS,
                  exhaustive: bool = False) -> dict:
    """Check a deterministic, stateless transform. Cut offsets and output rows are zero-based.

    Exhaustive mode checks every individual cut in ascending order until a failure.
    It cannot be combined with explicit cuts or a nondefault plan budget.
    """
    return _check(transform, frame, cuts, max_plans, exhaustive)


def check_elementwise(callback: Callable[[pl.Series], pl.Series], series: pl.Series, *,
                      cuts: Sequence[int] | None = None, max_plans: int = _DEFAULT_MAX_PLANS,
                      exhaustive: bool = False) -> dict:
    """Check whether a deterministic, pure Series callback is elementwise on this fixture.

    Each call must return one value per input row. Batched output must match the
    whole Series in dtype and exact ordered values. Output names are ignored.
    Split options match those of check_batches.
    """
    if not isinstance(series, pl.Series):
        raise TypeError("series must be a Polars Series")
    return _check(callback, series, cuts, max_plans, exhaustive, mode="series")


def check_expr_elementwise(expr: pl.Expr, frame: pl.DataFrame, *,
                           cuts: Sequence[int] | None = None, max_plans: int = _DEFAULT_MAX_PLANS,
                           exhaustive: bool = False) -> dict:
    """Check a deterministic expression using eager select on the whole frame and its slices.

    The expression must select one column and return one value per input row.
    Every batch must match the whole output dtype and exact ordered values.
    Split options match those of check_batches.
    """
    if not isinstance(expr, pl.Expr):
        raise TypeError("expr must be a Polars Expr")
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("frame must be a Polars DataFrame")

    def select(batch):
        # Stored row groups must not supply different batch boundaries to the baseline.
        out = batch.rechunk().select(expr)
        if out.width != 1:
            raise ValueError(f"expression must select exactly one output column (got {out.width})")
        return out

    return _check(select, frame, cuts, max_plans, exhaustive, mode="expr")
