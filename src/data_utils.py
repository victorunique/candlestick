"""Shared data utilities for the candlestick trading pipeline."""

import pandas as pd


def align_ticker_timestamps(
    df: pd.DataFrame,
    date_col: str = "date",
    tic_col: str = "tic",
) -> pd.DataFrame:
    """Filter a multi-ticker DataFrame to only keep timestamps present for ALL tickers.

    When intraday data is fetched for multiple tickers, each ticker may have
    slightly different available timestamps (trading halts, delayed opens, etc.).
    Pivoting such an unbalanced panel creates NaN for missing (date, tic) pairs,
    which cascades through portfolio value calculations.

    This function takes the **intersection** of timestamps across tickers,
    ensuring every timestamp has data for every ticker.

    Args:
        df: DataFrame with at least ``date_col`` and ``tic_col`` columns.
        date_col: Name of the timestamp/date column.
        tic_col: Name of the ticker column.

    Returns:
        Filtered DataFrame with only common timestamps, reset index.
    """
    if df.empty:
        return df

    tickers = df[tic_col].unique()
    if len(tickers) <= 1:
        return df

    # Build the intersection of timestamps across all tickers
    common_timestamps = None
    for tic in tickers:
        ts = set(df.loc[df[tic_col] == tic, date_col])
        common_timestamps = ts if common_timestamps is None else common_timestamps & ts

    if not common_timestamps:
        return df.iloc[0:0].reset_index(drop=True)  # empty with same schema

    return df[df[date_col].isin(common_timestamps)].reset_index(drop=True)
