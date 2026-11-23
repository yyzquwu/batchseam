# Batchseam

Does your Polars transform give the same answer when a batch ends halfway through the input? Batchseam runs the transform on the whole frame, splits the same rows into contiguous batches, and compares the concatenated results. A failure includes the cut offsets you can replay and the first differing output cell when both outputs have the same shape and schema.

## Try it

From this checkout, use a Python 3.11+ virtual environment.

```sh
python -m pip install .
python -m batchseam demo
```

The demo takes the running total of `[8, 3, 4, -2]`. It exits 1.

```text
FAIL  |  4 input rows  |  1 split plan checked
Replay this split with --cuts 2
First difference at output row 2 (zero-based), column 'balance'
Whole frame: 15
Batched:     4
```

## Check your own transform

Give Batchseam a trusted Python file with a function that accepts and returns an eager Polars DataFrame. It runs in the local Python process. Use a deterministic, stateless transform and a fixture small enough to fit in memory. Each test calls your function multiple times.

```sh
batchseam check examples/transforms.py:running_total examples/amounts.csv --cuts 2
batchseam check examples/transforms.py:cents examples/amounts.csv --json
```

CSV and Parquet inputs use the same command. The `cents` example passes because each row converts independently. A global running balance needs state across batches. In a real Polars pipeline, express that calculation with a native cumulative expression on the full lazy frame, or run the callback on the whole input.

```python
from batchseam import check_batches

result = check_batches(my_transform, frame, cuts=[2, 5])
assert result["status"] == "pass", result
```

The default tries a midpoint, a three-way split, edge cuts and additional interior offsets, stopping at the first difference or 16 plans. `--max-plans N` changes the budget. `--cuts 2,5` runs precisely the three slices `[:2]`, `[2:5]`, and `[5:]`. Comparisons check column names and order, types, row count, row order and values, including exact floating-point comparisons. An output offset refers to the transformed frame, which can have a different number of rows than the input.

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
