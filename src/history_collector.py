"""Continuously collect daily 1-minute OHLCV data from Yahoo Finance.

Run once per invocation — schedule with cron for periodic execution.
Checks the last N days (--lookback_days) and only fetches missing files.
"""

import argparse
import logging
import os
from datetime import date, timedelta

from src.data_fetcher import DataFetcher

logger = logging.getLogger(__name__)


def collect_day(ticker: str, target_date: date, data_dir: str) -> None:
    """Fetch 1-minute data for a single ticker and date, saving to CSV.

    Skips if the output file already exists.  Does not create empty files
    when Yahoo Finance returns no data (weekends / holidays).

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL").
        target_date: The calendar date to fetch.
        data_dir: Directory where CSV files are stored.
    """
    filename = f"{ticker}_{target_date.isoformat()}_1m.csv"
    filepath = os.path.join(data_dir, filename)

    if os.path.exists(filepath):
        logger.info("File already exists, skipping: %s", filename)
        return

    start_str = target_date.isoformat()
    end_str = (target_date + timedelta(days=1)).isoformat()

    try:
        fetcher = DataFetcher(
            start_date=start_str,
            end_date=end_str,
            ticker_list=[ticker],
            interval="1m",
        )
        fetcher.fetch_data(output_path=filepath)
        logger.info("Saved: %s", filename)
    except ValueError:
        # No data returned — expected on weekends / holidays.
        # Remove any partial file DataFetcher may have created.
        if os.path.exists(filepath):
            os.remove(filepath)
        logger.warning("No data for %s on %s (non-trading day?)", ticker, target_date)
    except Exception:
        if os.path.exists(filepath):
            os.remove(filepath)
        logger.exception("Failed to fetch %s for %s", ticker, target_date)


def run_cycle(tickers: list[str], data_dir: str, lookback_days: int = 7) -> None:
    """Run one collection cycle across all tickers and lookback dates.

    Args:
        tickers: List of ticker symbols.
        data_dir: Directory where CSV files are stored.
        lookback_days: Number of days to look back (including today).
    """
    today = date.today()
    dates = [today - timedelta(days=i) for i in range(lookback_days - 1, -1, -1)]

    logger.info(
        "Starting cycle: %d tickers × %d days (%s to %s)",
        len(tickers),
        len(dates),
        dates[0],
        dates[-1],
    )

    for target_date in dates:
        for ticker in tickers:
            collect_day(ticker, target_date, data_dir)

    logger.info("Cycle complete.")


def main() -> None:
    """CLI entry point — parse args, run one cycle, and exit."""
    parser = argparse.ArgumentParser(
        description="Collect daily 1-minute OHLCV data from Yahoo Finance."
    )
    parser.add_argument(
        "--tickers",
        type=str,
        nargs="+",
        required=True,
        help="List of ticker symbols (e.g. AAPL TSLA MSFT)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory to store CSV files (e.g. ./history)",
    )
    parser.add_argument(
        "--lookback_days",
        type=int,
        default=7,
        help="Number of days to look back, including today (default: 7)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    os.makedirs(args.data_dir, exist_ok=True)
    run_cycle(
        tickers=args.tickers,
        data_dir=args.data_dir,
        lookback_days=args.lookback_days,
    )


if __name__ == "__main__":
    main()
