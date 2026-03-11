import os
import pandas as pd
import tempfile
import pytest
import subprocess
import sys
from src.data_loader import load_local_history

def test_load_local_history():
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as out_dir:
        # Create dummy csv files
        df1 = pd.DataFrame({"date": ["2026-02-06 09:30:00-04:00"], "open": [1], "high": [1], "low": [1], "close": [1], "volume": [1], "tic": ["aapl"]})
        df1.to_csv(os.path.join(data_dir, "AAPL_2026-02-06_1m.csv"), index=False)
        
        df2 = pd.DataFrame({"date": ["2026-02-07 09:30:00-04:00"], "open": [2], "high": [2], "low": [2], "close": [2], "volume": [2], "tic": ["aapl"]})
        df2.to_csv(os.path.join(data_dir, "AAPL_2026-02-07_1m.csv"), index=False)
        
        output_path = os.path.join(out_dir, "out.csv")
        load_local_history(["AAPL"], "2026-02-06", "2026-02-08", data_dir, output_path)
        
        res = pd.read_csv(output_path)
        assert len(res) == 2
        assert res["close"].iloc[0] == 1
        assert res["close"].iloc[1] == 2

def test_load_local_history_empty():
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as out_dir:
        output_path = os.path.join(out_dir, "out.csv")
        with pytest.raises(ValueError, match="No local history data found"):
            load_local_history(["AAPL"], "2026-02-06", "2026-02-08", data_dir, output_path)

def test_data_loader_cli():
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as out_dir:
        df1 = pd.DataFrame({"date": ["2026-02-06 09:30:00-04:00"], "open": [1], "high": [1], "low": [1], "close": [1], "volume": [1], "tic": ["aapl"]})
        df1.to_csv(os.path.join(data_dir, "AAPL_2026-02-06_1m.csv"), index=False)

        output_path = os.path.join(out_dir, "cli_out.csv")
        
        cmd = [
            sys.executable, "-m", "src.data_loader",
            "--start_date", "2026-02-06",
            "--end_date", "2026-02-07",
            "--ticker_list", "AAPL",
            "--data_dir", data_dir,
            "--output_path", output_path
        ]
        
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0
        assert "Data successfully saved" in res.stdout
        
        assert os.path.exists(output_path)
        res_df = pd.read_csv(output_path)
        assert len(res_df) == 1
