from html import unescape

import pytest

from batchseam.summary import format_suite_markdown


def test_summary_counts_and_incomplete_checks():
    checks = [
        dict(name="scale", status="pass", report=dict(input_rows=4, checked=3)),
        dict(name="running total", status="fail", report=dict(
            input_rows=4, checked=1, cuts=[1, 3], difference=dict(kind="value", output_row=0))),
        dict(name="short fixture", status="insufficient", report=dict(input_rows=1, checked=0)),
        dict(name="broken transform", status="error", diagnostic="PRIVATE_TRACE"),
    ]
    md = format_suite_markdown(dict(version=1, status="error", checks=checks))

    assert "**Overall status** — **Error**" in md
    assert "4 checks · 1 pass · 1 fail · 1 insufficient · 1 error" in md
    assert "First differing output row (zero-based)" in md
    assert [line for line in md.splitlines() if line.startswith("| <code>")] == [
        "| <code>scale</code> | Pass | 4 | 3 | — | — | — |",
        "| <code>running total</code> | Fail | 4 | 1 | 1, 3 | Value mismatch | 0 |",
        "| <code>short fixture</code> | Insufficient | 1 | 0 | — | — | — |",
        "| <code>broken transform</code> | Error | — | — | — | — | — |",
    ]
    assert "fewer than two input rows" in md
    assert "did not produce a valid report" in md
    assert "PRIVATE_TRACE" not in md
    assert md.endswith("\n")


def test_summary_keeps_arbitrary_names_literal_in_one_cell():
    name = ("資料 🐍\r\n| **bold** _italics_ ~~strike~~ `code` \\ [x](https://example.com)"
            "\r</code><img src=x> &amp; @user :rocket:")
    md = format_suite_markdown(dict(version=1, status="pass", checks=[dict(name=name, status="pass")]))

    rows = [line for line in md.splitlines() if line.startswith("|")]
    assert len(rows) == 3
    assert all(row.count("|") == 8 for row in rows)
    cell = rows[-1].split("|")[1].strip()
    assert cell.startswith("<code>") and cell.endswith("</code>")
    encoded = cell.removeprefix("<code>").removesuffix("</code>")
    assert not any(char in encoded.replace("<br>", "") for char in "<>|*_~`\\[]()")
    assert unescape(encoded.replace("<br>", "\n")) == name.replace("\r\n", "\n").replace("\r", "\n")


@pytest.mark.parametrize(("kind", "label"), [
    ("row_count", "Row count mismatch"), ("length", "Length mismatch"),
    ("batch_error", "Batch error"), ("schema", "Schema mismatch"),
    (r"C:\PRIVATE_KIND", "—"),
])
def test_summary_only_includes_safe_report_fields(kind, label):
    report = dict(
        input_rows=2, checked=0, cuts=[], coverage={"fixture": r"C:\PRIVATE_FIXTURE.csv"},
        difference=dict(kind=kind, whole="PRIVATE_WHOLE", batched="PRIVATE_BATCH",
                        column="PRIVATE_COLUMN", input_values={"source": 314159}, error="PRIVATE_ERROR"),
    )
    check = dict(name="example", status="fail", report=report, archive=r"C:\PRIVATE_ARCHIVE")
    md = format_suite_markdown(dict(version=1, status="fail", checks=[check]))

    assert "**Overall status** — **Fail**" in md
    assert f"| Fail | 2 | 0 | Whole input | {label} | — |" in md
    assert "PRIVATE" not in md and "314159" not in md


def test_summary_omits_nonnumeric_split_and_row_metadata():
    check = dict(name="example", status="fail", report=dict(
        input_rows=2, checked=1, cuts=[1, "PRIVATE_CUT"],
        difference=dict(kind="value", output_row="PRIVATE_ROW"),
    ))
    md = format_suite_markdown(dict(version=1, status="fail", checks=[check]))

    assert "| Fail | 2 | 1 | — | Value mismatch | — |" in md
    assert "PRIVATE" not in md
