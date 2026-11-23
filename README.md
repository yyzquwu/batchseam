# Batchseam

**Find a batch boundary that changes your Polars transform, then keep the failing rows as a regression test.** Batchseam compares a whole-frame call with calls on contiguous slices of the same input.

## Try it

From this checkout, use a Python 3.12 virtual environment.

```sh
python -m pip install ".[test]"
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

Comparisons check column names and order, types, row count, row order and values, including exact floating-point comparisons. An output offset refers to the transformed frame, which can have a different number of rows than the input.

| Exit | Meaning |
| --- | --- |
| 0 | Every tested split matched on this fixture. Other inputs or splits may differ. |
| 1 | A batch changed the output or raised an exception after the whole-frame call succeeded. |
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
