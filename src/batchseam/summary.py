from collections import Counter
from string import punctuation


_STATUSES = ("pass", "fail", "insufficient", "error")
_KINDS = {
    "value": "Value mismatch",
    "row_count": "Row count mismatch",
    "length": "Length mismatch",
    "batch_error": "Batch error",
    "schema": "Schema mismatch",
}
_NAME_ESCAPES = str.maketrans({**{c: f"&#{ord(c)};" for c in punctuation}, "\n": "<br>"})
_MISSING = "—"


def _number(value):
    return str(value) if type(value) is int and value >= 0 else _MISSING


def format_suite_markdown(result: dict) -> str:
    """Format a suite result for a GitHub Actions step summary using safe report fields."""
    checks = result["checks"]
    counts = Counter(check["status"] for check in checks)
    totals = " · ".join(f"{counts[status]} {status}" for status in _STATUSES)
    lines = [
        "## Batchseam suite", "", f"**Overall status** — **{result['status'].capitalize()}**",
        "", f"{len(checks)} {'check' if len(checks) == 1 else 'checks'} · {totals}", "",
        "| Check | Status | Input rows | Split plans checked | Cuts | Failure kind | First differing output row (zero-based) |",
        "| --- | --- | ---: | ---: | --- | --- | ---: |",
    ]
    for check in checks:
        # Entities keep Markdown and table syntax literal. Code markup also suppresses autolinks.
        name = check["name"].replace("\r\n", "\n").replace("\r", "\n").translate(_NAME_ESCAPES)
        report = check.get("report") or {}
        cuts, diff = report.get("cuts"), report.get("difference") or {}
        cut_text = _MISSING
        if isinstance(cuts, list) and all(type(cut) is int and cut > 0 for cut in cuts):
            cut_text = ", ".join(map(str, cuts)) if cuts else "Whole input"
        kind = diff.get("kind")
        fields = (
            f"<code>{name}</code>", check["status"].capitalize(), _number(report.get("input_rows")),
            _number(report.get("checked")), cut_text, _KINDS.get(kind, _MISSING) if isinstance(kind, str) else _MISSING,
            _number(diff.get("output_row")),
        )
        lines.append(f"| {' | '.join(fields)} |")

    if counts["insufficient"]:
        lines.extend(("", "Insufficient checks had fewer than two input rows to split."))
    if counts["error"]:
        lines.extend(("", "Checks marked Error did not produce a valid report."))
    return "\n".join(lines) + "\n"
