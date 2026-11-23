import polars as pl


def running_total(df):
    return df.with_columns(balance=pl.col("amount").cum_sum())


def cents(df):
    return df.with_columns(cents=pl.col("amount") * 100)
