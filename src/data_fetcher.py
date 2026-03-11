import os
import pandas as pd
import yfinance as yf
from datetime import datetime

class DataFetcher:
    def __init__(self, start_date: str, end_date: str, ticker_list: list, interval: str = "1m"):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list
        self.interval = interval

    def fetch_data(self, output_path: str = None, auto_adjust: bool = False) -> pd.DataFrame:
        data_df = pd.DataFrame()
        num_failures = 0

        for tic in self.ticker_list:
            temp_df = yf.download(
                tic,
                start=self.start_date,
                end=self.end_date,
                interval=self.interval,
                auto_adjust=auto_adjust,
            )
            if len(temp_df) > 0:
                # Handle MultiIndex columns for single/multiple downloads in newer yfinance versions
                if temp_df.columns.nlevels != 1:
                    temp_df.columns = temp_df.columns.droplevel(1)
                
                temp_df["tic"] = tic.lower()
                data_df = pd.concat([data_df, temp_df], axis=0)
            else:
                num_failures += 1

        if num_failures == len(self.ticker_list):
            raise ValueError("no data is fetched.")

        # reset the index, we want dates/times as a column
        data_df = data_df.reset_index()
        
        # rename columns
        data_df.rename(
            columns={
                "index": "date",
                "Date": "date",
                "Datetime": "date",
                "Adj Close": "adjcp",
                "Close": "close",
                "High": "high",
                "Low": "low",
                "Volume": "volume",
                "Open": "open",
                "tic": "tic",
            },
            inplace=True,
        )

        if not auto_adjust and "adjcp" in data_df.columns:
            data_df = self._adjust_prices(data_df)
        elif "adjcp" in data_df.columns:
            # If auto_adjust=True, we still drop adjcp if it exists (though yf might not return it)
            data_df = data_df.drop(["adjcp"], axis=1)

        # format date column
        # user wants format similar to '2020-08-03 09:30:00-04:00'
        # ensure datetime timezone aware
        if data_df["date"].dt.tz is None:
            data_df["date"] = data_df["date"].dt.tz_localize("America/New_York")
        
        # Add timezone formatting. strftime("%Y-%m-%d %H:%M:%S%z") gives -0400, not -04:00
        # To get the colon, we can use isoformat and replace 'T' with space, or custom manipulation
        # Wait, if we use apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S%z")), we can manually add colon
        def format_date_with_colon(idx):
            s = idx.strftime("%Y-%m-%d %H:%M:%S%z")
            if s[-5:] == "+0000" or s[-5:] == "-0000" or len(s) < 5:
                # Some default handling for zero offset or missing
                pass
            return s[:-2] + ":" + s[-2:] if s[-5] in ["+", "-"] else s
            
        data_df["date"] = data_df["date"].apply(format_date_with_colon)

        # drop missing data
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)

        # sort values
        data_df = data_df.sort_values(by=["date", "tic"]).reset_index(drop=True)

        # write to CSV
        if output_path is None:
            output_path = f"./data_{int(datetime.now().timestamp())}.csv"
            
        # Ensure directory exists if there is one
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
        # Reorder columns to exactly match requiremets
        required_cols = ["date", "open", "high", "low", "close", "volume", "tic"]
        data_df = data_df[required_cols]

        data_df.to_csv(output_path, index=False)
        return data_df

    def _adjust_prices(self, data_df: pd.DataFrame) -> pd.DataFrame:
        data_df["adj"] = data_df["adjcp"] / data_df["close"]
        for col in ["open", "high", "low", "close"]:
            data_df[col] *= data_df["adj"]

        return data_df.drop(["adjcp", "adj"], axis=1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch historical stock data from Yahoo Finance.")
    parser.add_argument("--start_date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--ticker_list", type=str, nargs="+", required=True, help="List of ticker symbols")
    parser.add_argument("--output_path", type=str, default=None, help="Output CSV path (default: ./data_<timestamp>.csv)")
    parser.add_argument("--auto_adjust", action="store_true", help="Enable yfinance auto_adjust logic")
    parser.add_argument("--interval", type=str, default="1m", help="Data granularity (e.g., 1m, 1h, 1d)")

    args = parser.parse_args()

    fetcher = DataFetcher(
        start_date=args.start_date,
        end_date=args.end_date,
        ticker_list=args.ticker_list,
        interval=args.interval
    )
    
    print(f"Fetching data for {args.ticker_list} from {args.start_date} to {args.end_date}...")
    df = fetcher.fetch_data(output_path=args.output_path, auto_adjust=args.auto_adjust)
    print(f"Data saved successfully. Shape: {df.shape}")
