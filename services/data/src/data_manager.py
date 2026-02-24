import duckdb
import pandas as pd
from loguru import logger
import os

class DataManager:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.environ.get("DATABASE_PATH", "../../data_lake/market_data.duckdb")
        if self.db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self.conn = duckdb.connect(database=self.db_path)

    def setup_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                timestamp TIMESTAMP,
                symbol VARCHAR,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                PRIMARY KEY (symbol, timestamp)
            )
        """)

    def execute(self, query: str):
        return self.conn.execute(query)

    def ingest_dataframe(self, df: pd.DataFrame):
        cleaned_df = self.clean_and_impute(df)
        self.conn.execute("""
            INSERT INTO market_data
            SELECT * FROM cleaned_df
            ON CONFLICT (symbol, timestamp) DO UPDATE
            SET open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low, close=EXCLUDED.close, volume=EXCLUDED.volume
        """)

    def get_sliding_window(self, symbol: str, timestamp, window_size: int = 1000) -> pd.DataFrame:
        query = f"""
            SELECT * FROM market_data
            WHERE symbol = ? AND timestamp <= ?
            ORDER BY timestamp DESC
            LIMIT ?
        """
        df = self.conn.execute(query, [symbol, timestamp, window_size]).df()
        
        if len(df) < window_size:
            raise ValueError(f"Not enough data points. Requested {window_size}, got {len(df)}")
            
        return df.sort_values("timestamp").reset_index(drop=True)

    def clean_and_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
            
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        dfs = []
        for sym, group in df.groupby("symbol"):
            group = group.set_index("timestamp").sort_index()
            # resample to '1min' internally uses 'min' in newer pandas
            resampled = group.resample("min").asfreq()
            
            # forward-fill close price
            resampled["close"] = resampled["close"].ffill()
            for col in ["open", "high", "low"]:
                resampled[col] = resampled[col].fillna(resampled["close"])
            
            resampled["volume"] = resampled["volume"].fillna(0)
            resampled["symbol"] = sym
            resampled = resampled.reset_index()
            dfs.append(resampled)
            
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return df
