# Batchseam 1.x compatibility

Batchseam checks deterministic Python and Polars code against explicit, contiguous row slices. Results describe the supplied fixture and the split plans actually evaluated. The CLI executes project code only for checks and replay. `scan` reads Python syntax.

## Python API

- `check_batches(function, frame)` compares a DataFrame transform on the whole frame and on concatenated transformed slices. It permits changes in row count when the two executions agree.
- `check_elementwise(function, series)` requires a Polars Series with one result per input row on the whole fixture and in every tested batch.
- `check_expr_elementwise(expr, frame)` evaluates an existing Polars expression with eager `select`. It expects a single output column, one value per input row, and agreement on dtype and ordered values. Physical chunks are consolidated before selecting.
- `assert_batch_invariant`, `assert_elementwise`, and `assert_expr_elementwise` accept the corresponding inputs. They return `None` on a pass and raise `AssertionError` on a demonstrated failure or an insufficient fixture. Invalid inputs, whole-input execution errors and unsupported comparisons can raise an exception.
- Each checker accepts an explicit `cuts` sequence, a `max_plans` budget, or `exhaustive=True` for single cuts. Cut offsets, source spans and differing output rows are zero-based. A span's end is exclusive. The default planner can evolve in 1.x. Pass explicit cuts or replay a bundle to repeat a particular split.

## CLI and JSON

`demo`, `check`, and `replay` return 0 for pass, 1 for a demonstrated difference, and 2 for insufficient data or an error. With `--json`, a completed comparison writes one report to stdout with `status`, `input_rows`, and `checked`. Failures add `cuts` and `difference`. A whole-input length failure can have no cuts. Representable value differences include the whole and batched values. For expression witnesses, a non-null input value with no lossless supported JSON representation is omitted from `input_values`. A configuration or execution error can exit 2 with its diagnostic on stderr and no report.

`scan` exits 0 for a complete static scan, including scans with findings. An incomplete scan exits 2 and can still return findings and errors together. A finding identifies a claim to test. Dynamic code can prevent identification of a callback.

`suite` reads manifest version 1. A valid manifest runs every named check in declaration order. Outcomes are `pass`, `fail`, `insufficient`, or `error`, with the overall precedence `error`, `insufficient`, `fail`, `pass`. The two incomplete outcomes exit 2. A malformed manifest exits 2 before the suite runs. Suite JSON has `version`, `status`, and `checks`. Successful launches with a comparison report retain it, and runtime errors include a diagnostic. Consumers should accept additional fields in 1.x reports. Human-readable messages and Markdown layout may change.

## Saved failures

DataFrame, Series, and expression bundles use manifest versions 1, 2, and 3 respectively. The v1.0 CLI reads all three. Each bundle includes the fixture, saved cuts, source reference and a pytest regression. Replay executes the current source in the project environment. A source edit can turn a red test green. Keep a bundle at its relative location when moving the checkout; on different Windows drives a source reference can be absolute.

`suite --save-failures DIR` requires a new output directory. Numbered bundles use manifest check order. A completed run writes `summary.json` even if every check passed. A missing summary can indicate an interrupted run. JSON reports and Parquet fixtures can contain original data. Select artifact storage and sharing permissions accordingly.
