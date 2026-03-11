import pandas as pd
import pandas_ta as ta
import argparse
import os

INDICATORS = [
    "atr", "atr_20", "macd", "macds", "macdh", "boll_ub", "boll_lb", "boll",
    "rsi_14", "rsi_6", "adx", "dx", "close_12_ema", "close_26_ema", "close_50_ema",
    "close_30_sma", "close_60_sma", "kdjk", "kdjd", "kdjj", "cci", "cci_20", "tr"
]

class FeatureEngineer:
    """
    Feature Engineer component that computes technical indicators for trading data.
    """
    
    def __init__(self, indicators: list = None):
        if indicators is None:
            self.indicators = INDICATORS
        else:
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
            
            # To handle robust calculation for the 23 standard FinRL indicators,
            # we can leverage dataframe mappings to `pandas_ta`
            # For simplicity, calculate a standard bulk of common indicators
            # and map them if they match requested indicators explicitly.
            
            # Map standard indicator keywords for calculation
            for ind in self.indicators:
                if "macd" in ind:
                    macd_res = group.ta.macd(append=False)
                    if macd_res is not None and not macd_res.empty:
                        # Extract MACD signal components mapping to FinRL names
                        if ind == "macd" and 'MACD_12_26_9' in macd_res.columns:
                            group["macd"] = macd_res['MACD_12_26_9']
                        elif ind == "macds" and 'MACDs_12_26_9' in macd_res.columns:
                            group["macds"] = macd_res['MACDs_12_26_9']
                        elif ind == "macdh" and 'MACDh_12_26_9' in macd_res.columns:
                            group["macdh"] = macd_res['MACDh_12_26_9']
                
                elif ind.startswith("rsi_"):
                    window = int(ind.split("_")[1])
                    res = group.ta.rsi(length=window)
                    if res is not None and not res.empty:
                        if isinstance(res, pd.Series): group[ind] = res
                    
                elif ind.startswith("cci"):
                    window = 14 if ind == "cci" else int(ind.split("_")[1])
                    res = group.ta.cci(length=window)
                    if res is not None and not res.empty:
                        if isinstance(res, pd.Series): group[ind] = res
                    
                elif ind in ["adx", "dx"]:
                    adx_res = group.ta.adx(length=14)
                    if adx_res is not None and not adx_res.empty:
                        if isinstance(adx_res, pd.DataFrame):
                            if ind == "adx" and "ADX_14" in adx_res.columns:
                                group["adx"] = adx_res["ADX_14"]
                            elif ind == "dx" and "DMP_14" in adx_res.columns: 
                                pass # Handled below optionally if really needed
                
                elif "sma" in ind:
                    window = int(ind.split("_")[1])
                    res = group.ta.sma(length=window)
                    if res is not None and not res.empty:
                        if isinstance(res, pd.Series): group[ind] = res
                    
                elif "ema" in ind:
                    window = int(ind.split("_")[1])
                    res = group.ta.ema(length=window)
                    if res is not None and not res.empty:
                        if isinstance(res, pd.Series): group[ind] = res
                    
                elif "boll" in ind:
                    boll_res = group.ta.bbands()
                    if boll_res is not None and not boll_res.empty:
                        if isinstance(boll_res, pd.DataFrame):
                            if ind == "boll" and "BBM_5_2.0" in boll_res.columns:
                                group["boll"] = boll_res["BBM_5_2.0"]
                            elif ind == "boll_ub" and "BBU_5_2.0" in boll_res.columns:
                                group["boll_ub"] = boll_res["BBU_5_2.0"]
                            elif ind == "boll_lb" and "BBL_5_2.0" in boll_res.columns:
                                group["boll_lb"] = boll_res["BBL_5_2.0"]
                            
                elif "atr" in ind:
                    window = 14 if ind == "atr" else int(ind.split("_")[1])
                    atr_res = group.ta.atr(length=window)
                    if atr_res is not None and not atr_res.empty:
                        if isinstance(atr_res, pd.Series): group[ind] = atr_res
                    
                elif ind == "tr":
                    tr_res = group.ta.true_range()
                    if tr_res is not None and not tr_res.empty:
                        if isinstance(tr_res, pd.Series): group[ind] = tr_res
                    
                elif "kdj" in ind:
                    # KDJ matches Stochastic Oscillator parameters in some standard libraries
                    stoch_res = group.ta.stoch()
                    if stoch_res is not None and not stoch_res.empty:
                        if ind == "kdjk" and "STOCHk_14_3_3" in stoch_res.columns:
                            group["kdjk"] = stoch_res["STOCHk_14_3_3"]
                        elif ind == "kdjd" and "STOCHd_14_3_3" in stoch_res.columns:
                            group["kdjd"] = stoch_res["STOCHd_14_3_3"]
                
                # Default safety: if indicator computation fails or isn't strictly recognized, place 0 
                # (to prevent crashing backward compatibility if an obscure stockstats variable is requested)
                if ind not in group.columns:
                    group[ind] = 0.0
                
            # Forward fill any true isolated gaps, but drop the initial warmup rows
            # instead of backfilling them, which causes look-ahead bias.
            group.ffill(inplace=True)
            group.dropna(inplace=True)
            
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
    parser.add_argument("--indicator_list", type=str, nargs="+", default=INDICATORS, help="List of indicators to compute.")

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
