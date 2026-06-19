import pytest
import pandas as pd
import tempfile
import os
from src.csv_to_latex import escape_latex, format_value, generate_latex_table

def test_escape_latex():
    # Test escaping of LaTeX special characters
    assert escape_latex("Hello_World") == r"Hello\_World"
    assert escape_latex("50%") == r"50\%"
    assert escape_latex("A & B") == r"A \& B"
    assert escape_latex("Price $10") == r"Price \$10"
    assert escape_latex("a # b") == r"a \# b"
    assert escape_latex("a { b } c") == r"a \{ b \} c"
    assert escape_latex("~tilde") == r"\textasciitilde tilde"
    assert escape_latex("^caret") == r"\textasciicircum caret"
    assert escape_latex("back\\slash") == r"back\textbackslash slash"

def test_format_value():
    # Test formatting floats
    assert format_value(123.4567, "some_float", precision=2) == "123.46"
    assert format_value(123.4567, "some_float", precision=4) == "123.4567"
    
    # Test formatting integers (with comma formatting)
    assert format_value(1234567, "volume", precision=2) == "1,234,567"
    assert format_value(123, "volume", precision=2) == "123"
    
    # Test datetime formatting
    assert format_value("2026-06-18 13:30:00+00:00", "date", precision=2) == "2026-06-18 13:30:00"
    
    # Test handling NaN/None/Null values
    assert format_value(None, "col", precision=2) == "-"
    assert format_value(float('nan'), "col", precision=2) == "-"

def test_generate_latex_table_basic():
    # Create a small dummy dataframe
    data = {
        "date": ["2026-06-18 13:30:00+00:00"],
        "close": [299.67999267578125],
        "volume": [11120403],
        "tic": ["aapl"]
    }
    df = pd.DataFrame(data)
    
    # Default generate table
    latex = generate_latex_table(df, precision=2, bold_headers=False, booktabs=False)
    
    # Assert basics
    assert r"\begin{tabular}" in latex
    assert "date & close & volume & tic" in latex
    # Check values
    assert "2026-06-18 13:30:00" in latex
    assert "299.68" in latex
    assert "11,120,403" in latex
    assert "aapl" in latex
    assert r"\hline" in latex
    assert r"\end{tabular}" in latex

def test_generate_latex_table_booktabs_and_headers():
    data = {
        "date": ["2026-06-18 13:30:00+00:00"],
        "close": [299.67999267578125],
        "tic": ["aapl"]
    }
    df = pd.DataFrame(data)
    
    # Custom mapping, bold headers, booktabs
    columns_map = {"date": "Timestamp", "close": "Close Price", "tic": "Ticker"}
    latex = generate_latex_table(
        df, 
        columns_map=columns_map, 
        precision=2, 
        bold_headers=True, 
        booktabs=True,
        caption="Sample Stock Data",
        label="tab:sample_stock"
    )
    
    assert r"\begin{table}" in latex
    assert r"\centering" in latex
    assert r"\caption{Sample Stock Data}" in latex
    assert r"\label{tab:sample_stock}" in latex
    assert r"\toprule" in latex
    assert r"\midrule" in latex
    assert r"\bottomrule" in latex
    assert r"\textbf{Timestamp} & \textbf{Close Price} & \textbf{Ticker}" in latex

def test_generate_latex_table_standalone():
    data = {
        "tic": ["aapl"]
    }
    df = pd.DataFrame(data)
    latex = generate_latex_table(df, standalone=True)
    
    assert r"\documentclass" in latex
    assert r"\begin{document}" in latex
    assert r"\end{document}" in latex

def test_main_cli_execution():
    from unittest.mock import patch
    import sys
    from src.csv_to_latex import main
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test.csv")
        tex_path = os.path.join(tmpdir, "test.tex")
        
        # Write dummy CSV
        df = pd.DataFrame({"col1": [1.234, 5.678], "col2": [10, 20]})
        df.to_csv(csv_path, index=False)
        
        test_args = [
            "csv_to_latex.py",
            "-i", csv_path,
            "-o", tex_path,
            "-c", "col1:Col A,col2:Col B",
            "--precision", "1",
            "--caption", "Test Table",
            "--label", "tab:test"
        ]
        
        with patch.object(sys, 'argv', test_args):
            exit_code = main()
            assert exit_code == 0
            
        # Verify content written
        assert os.path.exists(tex_path)
        with open(tex_path, "r") as f:
            content = f.read()
            assert r"\caption{Test Table}" in content
            assert r"\label{tab:test}" in content
            # col1 should be rounded to 1 decimal place -> 1.2 and 5.7
            assert "1.2 & 10" in content
            assert "5.7 & 20" in content

