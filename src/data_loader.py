import os
import pandas as pd
from datetime import datetime, timedelta

def load_local_history(tickers: list[str], start_date: str, end_date: str, data_dir: str, output_path: str) -> None:
    """Load and merge 1-minute historical ticker data from a local folder.
    
    Like Yahoo Finance, dates are inclusive of start_date and exclusive of end_date.
    
    Args:
        tickers: List of ticker symbols to load data for.
        start_date: Start date string in YYYY-MM-DD format (inclusive).
        end_date: End date string in YYYY-MM-DD format (exclusive).
        data_dir: Path to directory containing the local CSV history files.
        output_path: Path where the merged output CSV should be saved.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()

    dates = []
    curr = start_dt
    while curr < end_dt:
        dates.append(curr)
        curr += timedelta(days=1)

    df_list = []
    for tic in tickers:
        for d in dates:
            filename = f"{tic}_{d.isoformat()}_1m.csv"
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath, dtype={"date": str})
                    if not df.empty:
                        df_list.append(df)
                except Exception as e:
                    print(f"Warning: failed to read {filepath}: {e}")

    if not df_list:
        raise ValueError(f"No local history data found for {tickers} from {start_date} to {end_date} in {data_dir}")

    final_df = pd.concat(df_list, axis=0, ignore_index=True)
    if "date" in final_df.columns:
        final_df = final_df.sort_values(by=["date", "tic"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    required_cols = ["date", "open", "high", "low", "close", "volume", "tic"]
    existing_cols = [c for c in required_cols if c in final_df.columns]
    final_df = final_df[existing_cols]

    final_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Manually create training and testing datasets from a local historical folder.")
    parser.add_argument("--start_date", type=str, required=True, help="Start date (YYYY-MM-DD), inclusive.")
    parser.add_argument("--end_date", type=str, required=True, help="End date (YYYY-MM-DD), exclusive.")
    parser.add_argument("--ticker_list", type=str, nargs="+", required=True, help="List of ticker symbols")
    parser.add_argument("--data_dir", type=str, default="./history", help="Data directory (default: ./history)")
    parser.add_argument("--output_path", type=str, required=True, help="Output CSV path")

    args = parser.parse_args()

    print(f"Loading local data for {args.ticker_list} from {args.start_date} to {args.end_date}...")
    try:
        load_local_history(
            tickers=args.ticker_list,
            start_date=args.start_date,
            end_date=args.end_date,
            data_dir=args.data_dir,
            output_path=args.output_path
        )
        print(f"Data successfully saved to {args.output_path}")
    except Exception as e:
        print(f"Error: {e}")
