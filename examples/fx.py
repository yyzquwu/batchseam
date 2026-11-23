"""Compare a batch-dependent FX calculation with a per-row conversion."""

import polars as pl


def _first_rate(columns):
    amount, rate = columns
    return amount * rate.first()


usd = pl.map_batches(["amount", "rate"], _first_rate, return_dtype=pl.Float64, is_elementwise=True).alias("usd")
usd_fixed = (pl.col("amount") * pl.col("rate")).alias("usd")
