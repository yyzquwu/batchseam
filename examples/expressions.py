"""Compare the one-result-per-row contract with a native Polars expression."""

import polars as pl


def keep_positive(values):
    return values.filter(values > 0)


if __name__ == "__main__":
    from batchseam import check_batches, check_elementwise

    df = pl.DataFrame({"amount": [-1, 2, 3]})
    frame_check = check_batches(lambda b: keep_positive(b["amount"]).to_frame(), df, exhaustive=True)
    series_check = check_elementwise(keep_positive, df["amount"])
    print("DataFrame split check:", frame_check)
    print("Row-aligned Series check:", series_check)
    try:
        df.with_columns(out=pl.col("amount").map_batches(keep_positive, return_dtype=pl.Int64, is_elementwise=True))
    except pl.exceptions.ShapeError as err:
        print("Native Polars:", err)
