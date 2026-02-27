import pandas as pd
import pandas_ta as ta
import argparse
import os

class FeatureEngineer:
    """
    Feature Engineer component that computes technical indicators for trading data.
    """
    
    def __init__(self, indicators: list):
        self.indicators = indicators
        self.required_columns = ["date", "tic", "open", "high", "low", "close", "volume"]

    def _check_columns(self, df: pd.DataFrame):
        missing = [col for col in self.required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in input data: {missing}")

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the specified indicators for each ticker in the dataframe.
        """
        self._check_columns(df)
        
        # Calculate indicators group by ticker
        def add_indicators(group):
            # Sort by date just to be safe
            group = group.sort_values("date")
            
            # Use pandas_ta for indicator calculations
            # To handle multiple indicators, we can use the DataFrame extension `ta`
            # or calculate explicitly.
            
            # Mapping our simple indicator names to pandas_ta calls
            for ind in self.indicators:
                if ind == "macd":
                    # macd returns MACD, MACDh, MACDs. We just want MACD signal or main line. Let's take MACD
                    macd = group.ta.macd(append=False)
                    if macd is not None and not macd.empty:
                        # Extract the MACD line (usually MACD_12_26_9)
                        group["macd"] = macd.iloc[:, 0]
                elif ind.startswith("rsi_"):
                    window = int(ind.split("_")[1])
                    group[ind] = group.ta.rsi(length=window)
                elif ind.startswith("cci_"):
                    window = int(ind.split("_")[1])
                    group[ind] = group.ta.cci(length=window)
                elif ind.startswith("dx_"):
                    window = int(ind.split("_")[1])
                    # Note: ADX calculation usually returns ADX, DMP, DMN. dx is specifically directional movement index
                    # Let's use adx and take the DX column if available or ADX
                    adx = group.ta.adx(length=window)
                    if adx is not None and not adx.empty:
                        # Depending on pandas_ta version, finding the right column
                        # ADX_14, DMP_14, DMN_14. DX might need explicit calculation, but ADX is often used interchangeably in FinRL
                        group[ind] = adx.iloc[:, 0]  # Take ADX line
                elif ind.startswith("sma_"):
                    window = int(ind.split("_")[1])
                    group[ind] = group.ta.sma(length=window)
                
            # Fill missing values created by rolling windows with backfill then forward fill
            group.bfill(inplace=True)
            group.ffill(inplace=True)
            group.fillna(0, inplace=True)
            
            return group
            
        # Apply the calculations per ticker
        # Iterate over groups and concatenate to avoid pandas FutureWarning on groupby.apply
        processed_df = pd.concat([add_indicators(group.copy()) for _, group in df.groupby("tic")])
        
        # Re-sort to match original finrl date-first structure
        processed_df = processed_df.sort_values(by=["date", "tic"]).reset_index(drop=True)
        return processed_df

def main():
    parser = argparse.ArgumentParser(description="Feature Engineer: Compute technical indicators for dataset.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input raw CSV data.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save preprocessed CSV data.")
    parser.add_argument("--indicator_list", type=str, nargs="+", default=["macd", "rsi_30", "cci_30", "dx_30"], help="List of indicators to compute.")

    args = parser.parse_args()

    print(f"Loading data from {args.input_path}...")
    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input file {args.input_path} not found.")

    df = pd.read_csv(args.input_path)
    print(f"Loaded {len(df)} rows. Computing technical indicators: {args.indicator_list}")

    fe = FeatureEngineer(indicators=args.indicator_list)
    processed_df = fe.preprocess_data(df)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    
    print(f"Saving preprocessed data to {args.output_path}...")
    processed_df.to_csv(args.output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
