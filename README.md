# Batchseam

**Check a Polars batch callback and keep the failing rows as a pytest regression.** For an elementwise Series callback, Batchseam checks that every input row gets one output and that changing the batch boundary preserves dtype, order, and values. For a DataFrame transform, it checks whether concatenating the transformed slices reproduces the whole-frame result.

## Find your first check

From this checkout, use a Python 3.12 virtual environment.

```sh
python -m pip install ".[test]"
batchseam scan .
```

Run `scan` in a Python project to find calls to `map_batches` with an explicit `is_elementwise=True` or `streamable=True`. Each finding shows a source location, the callback, and a suggested next step. The scanner reads Python syntax without importing your project. A finding identifies a claim to test. You supply the fixture and run the check to learn whether the claim holds.

Try the whole workflow from the Batchseam checkout:

```sh
cd examples
batchseam scan .
batchseam check expressions.py:keep_positive amounts.csv --series amount --save-repro positive-repro
batchseam replay positive-repro
python -m pytest -q positive-repro
```

The scan exits 0. The check exits 1 because `amounts.csv` contains four amounts and the callback returns only three positive ones. Replay and pytest reproduce the failure. When trying your own project, substitute a representative CSV or Parquet file and the actual input column in the command template. Relative commands use the root printed by the scan. Start subsequent examples from the checkout root.

### How to use a finding

- **Named function.** SportsDataverse calls `expr.map_batches(_series_as_date, ..., is_elementwise=True)` on an annotated Polars expression. The scan can point to the function in the same file. On a three-row fixture of ISO dates and a null, this callback passed both individual cuts. [Upstream date parser](https://github.com/sportsdataverse/sportsdataverse-py/blob/2443d9ffd8f5a493d189079814fef8d33f767bf0/sportsdataverse/_temporal.py#L13-L19)
- **Inline lambda.** Extract it into a shared named callback so your pipeline and test execute the same function. [Geometry conversion example](https://github.com/oreilles/polars-st/blob/58dca7a693b7f005fbe031ae39160cae7c85760d/polars_st/parsing.py#L375-L384)
- **Bound method.** Construct the instance with your project's configuration and pass its bound method to the assertion. [Interpolation example](https://github.com/crdcj/PYield/blob/5572bf034227fd782f70f6b6d0ea4f389a40397f/pyield/interpolador.py#L293)
- **Module-qualified function.** For a callback such as `helpers.normalize`, import the same module in your test and pass `helpers.normalize` directly to the assertion.

An **ambiguous** receiver needs a human to confirm its type. Dynamic flags, wrappers and cross-module resolution can leave calls undiscovered. For automation, use `batchseam scan . --json`. An incomplete scan exits 2 and reports the files that could not be read or parsed alongside findings from the other files.

You can also scan one Python file. A directory scan skips nested environments, dependency and build folders, links, and recognized generated files. Selecting a folder such as `build` explicitly scans its source. Selecting a virtual environment itself reports an incomplete scan.

## Try the Series demos

```sh
python -m batchseam demo dates
python -m batchseam demo center
python -m batchseam demo positive
```

| Series callback | What the fixture shows | Exit |
| --- | --- | --- |
| Parse ISO dates | Dates and nulls stay in the same positions across the tested splits. | 0 |
| Subtract the batch mean | For `[10, 30]`, the first output changes from −10 to 0 when the values are processed separately. | 1 |
| Keep positive values | Three inputs produce only two outputs, before a split is even tried. | 1 |

## Test an elementwise callback

```python
import polars as pl
from batchseam import assert_elementwise

def parse_date(values):
    return values.str.to_date("%Y-%m-%d")

def test_parse_date():
    dates = pl.Series("day", ["2026-01-01", None, "2026-01-03"])
    assert_elementwise(parse_date, dates)
```

Pass the same pure, single-input `Series -> Series` function you would give to `Expr.map_batches`. The callback may change dtype, as the date example does. Its results must keep that output dtype when the input is split. Use `check_elementwise` to get the result dictionary. A passing fixture is evidence for the cuts tried. [Polars expression callback contract](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.map_batches.html)

```sh
batchseam check examples/expressions.py:keep_positive examples/amounts.csv --series amount --save-repro positive-repro
batchseam replay positive-repro --json
python -m pytest -q positive-repro
```

These commands intentionally fail on the sample callback. The Parquet bundle contains only the selected column and replays the Series check against the current callback. Use at least two rows for a regression test you expect to pass after a fix. A shorter fixture can demonstrate a length violation, but after correction there is no interior cut to test and replay returns exit 2. Existing v0.2 DataFrame bundles still replay.

Run `python examples/expressions.py` to see the distinction on `[-1, 2, 3]`: filtering commutes with every single split, but native Polars cannot place the two surviving values in a three-row column.

## Find a boundary in a DataFrame transform

```sh
python -m batchseam demo sensor --save-repro sensor-repro
```

The demo forward-fills 100 sensor readings. Only row 41 is null. When a batch starts there, the callback loses the preceding value. The demo exits 1 and saves the failure in a new directory.

| Reduced input | Whole-frame output | Output with a cut after the first row |
| --- | --- | --- |
| `[40, null]` | `[40, 40]` | `[40, null]` |

```sh
python -m batchseam replay sensor-repro
python -m pytest -q sensor-repro
```

Both commands reproduce the failure. The bundle contains the ordered rows in `fixture.parquet`, the transform reference and cut offsets in `repro.json`, and a pytest test. For a transform from your project, the test runs your current code. Keep the bundle and source in the same relative locations when moving a project. Paths across Windows drives use an absolute reference. The original running-total example is still available with `python -m batchseam demo`.

## Check your own transform

Give Batchseam a trusted Python file with a function that accepts and returns an eager Polars DataFrame. Regular Python packages can use package-relative imports. Make your project's other dependencies available in the active environment, just as you would for its tests. The generated test passes pytest's import paths to the replay. Use a deterministic, stateless transform and a fixture small enough to fit in memory. Each test calls your function multiple times.

```sh
batchseam check examples/transforms.py:running_total examples/amounts.csv --cuts 2 --save-repro balance-repro
batchseam check examples/transforms.py:cents examples/amounts.csv --json
```

CSV and Parquet inputs use the same command. The `cents` example passes because each row converts independently. A global running balance needs state across batches. In a real Polars pipeline, express that calculation with a native cumulative expression on the full lazy frame, or run the callback on the whole input.

```python
from batchseam import assert_batch_invariant

def test_my_transform(frame):
    assert_batch_invariant(my_transform, frame, cuts=[2, 5])
```

Use `check_batches` when you want the result dictionary. For either API, the default mixes ordinary cuts with boundaries where columns enter or leave null runs. It stops at the first failure or 16 plans. On the sensor fixture, v0.1 passed all 16 default plans and needed 48 attempts with an increased budget to find cut 41. Version 0.2 finds it on attempt five. Checking all 99 single cuts independently confirmed that only cut 41 fails.

- `--max-plans N` sets the discovery budget.
- `--exhaustive` tries every single interior cut in order until one fails. Use it on small fixtures. Combinations of multiple cuts can expose additional failures.
- `--cuts 2,5` tests precisely `[:2]`, `[2:5]`, and `[5:]`.
- `--save-repro DIR` tries up to 128 reductions and rechecks the saved Parquet data. It can return a larger fixture when further reduction changes the failure. The rows still contain their original values. Choose a new directory and review the rows before sharing it. A successful check creates no bundle.

Comparisons use native Polars dtypes and exact floating-point comparisons. DataFrame checks also compare column names and order. An output offset refers to the transformed frame, which can have a different number of rows than the input. If comparing values requires Python `Object` equality, the CLI reports an inconclusive run (exit 2). Use a native output dtype, such as a string representation of a UUID, for an ordered value comparison.

| Exit | Meaning |
| --- | --- |
| 0 | Every tested split matched on this fixture. Other inputs or splits may differ. |
| 1 | The callback broke the Series length contract, a batch changed the output, or a batch raised after the whole-input call succeeded. |
| 2 | The run was inconclusive. This includes fewer than two rows, an invalid command or a failed whole-frame call. |

## Native streaming experiment

```sh
python examples/streaming.py
```

Polars requires a streamable `map_batches` callback to produce the same result on batches as on the full dataset. Its streaming design discussion uses cumulative sums to illustrate the need for cross-batch state. [Polars documentation](https://docs.pola.rs/api/python/stable/reference/lazyframe/api/polars.LazyFrame.map_batches.html) · [Polars discussion](https://github.com/pola-rs/polars/issues/25786)

The script marks the two example callbacks streamable and lets native Polars choose its own batches. On one Windows run with Polars 2.1.0, eight batches of 100,000 total one-unit rows yielded a final running balance of 12,500 versus 100,000 on the whole frame. The row-local conversion finished at 100 cents in both runs. Batch boundaries depend on the environment. Batchseam's explicit-cut replay is independent of the native schedule.

## Run the tests

```sh
python -m pip install ".[test]"
python -m pytest -q
```
