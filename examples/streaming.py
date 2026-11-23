"""Run the same callbacks eagerly, through native Polars streaming, and with Batchseam."""

import polars as pl

from batchseam import check_batches
from transforms import cents, running_total


def compare(df, fn, column):
    sizes = []

    def measured(batch):
        sizes.append(batch.height)
        return fn(batch)

    whole = fn(df)
    streamed = df.lazy().map_batches(measured, schema=whole.schema, streamable=True).collect(engine="streaming")
    result = check_batches(fn, df, cuts=[df.height // 2])
    print(f"{fn.__name__}: whole={whole[column][-1]}, streaming={streamed[column][-1]}, "
          f"native_batches={len(sizes)}, batchseam={result['status']}")


if __name__ == "__main__":
    df = pl.DataFrame({"amount": [1] * 100_000})
    print(f"Polars {pl.__version__}, {df.height} rows")
    compare(df, running_total, "balance")
    compare(df, cents, "cents")
