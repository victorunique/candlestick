import pandas as pd
import numpy as np

def escape_latex(text: str) -> str:
    """
    Escapes LaTeX special characters in a text string.
    """
    if not isinstance(text, str):
        text = str(text)
    # Escape backslash first to prevent double-escaping
    text = text.replace('\\', '\\textbackslash ')
    # Escape other special characters
    text = text.replace('~', '\\textasciitilde ')
    text = text.replace('^', '\\textasciicircum ')
    for char in ['&', '%', '$', '#', '_', '{', '}']:
        text = text.replace(char, '\\' + char)
    return text

def format_value(val, col_name: str, precision: int = 2) -> str:
    """
    Formats a single cell value for LaTeX table display depending on its type and column name.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)) or pd.isna(val):
        return "-"
    
    col_lower = col_name.lower()
    
    # Datetime formatting
    if "date" in col_lower or "time" in col_lower:
        try:
            dt = pd.to_datetime(val)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
            
    # Volume formatting (integers with commas)
    if "volume" in col_lower:
        try:
            return f"{int(float(val)):,}"
        except (ValueError, TypeError):
            pass
            
    # Standard integers formatting (with commas)
    if isinstance(val, (int, np.integer)):
        return f"{val:,}"
        
    # Floats formatting (with precision control)
    if isinstance(val, (float, np.floating)):
        return f"{val:.{precision}f}"
        
    # If it's a string, try parsing it to number/float first if it looks numeric
    if isinstance(val, str):
        try:
            if val.isdigit():
                return f"{int(val):,}"
            f_val = float(val)
            if f_val.is_integer():
                return f"{int(f_val):,}"
            return f"{f_val:.{precision}f}"
        except ValueError:
            pass
            
    return escape_latex(str(val))

def generate_latex_table(
    df: pd.DataFrame, 
    columns_map: dict = None, 
    precision: int = 2, 
    bold_headers: bool = True, 
    booktabs: bool = True, 
    caption: str = None, 
    label: str = None, 
    alignment: str = None, 
    standalone: bool = False
) -> str:
    """
    Generates LaTeX table code from a pandas DataFrame.
    """
    if columns_map:
        cols = [c for c in columns_map.keys() if c in df.columns]
        df_subset = df[cols].copy()
        display_headers = [columns_map[c] for c in cols]
    else:
        df_subset = df.copy()
        display_headers = list(df.columns)
        
    # Format headers
    if bold_headers:
        formatted_headers = [r"\textbf{" + escape_latex(h) + "}" for h in display_headers]
    else:
        formatted_headers = [escape_latex(h) for h in display_headers]
        
    # Column alignment auto-detection if not specified
    if not alignment:
        align_list = []
        for col in df_subset.columns:
            if pd.api.types.is_numeric_dtype(df_subset[col]):
                align_list.append("r")
            else:
                align_list.append("l")
        alignment = "".join(align_list)
        
    # Format rows
    rows_latex = []
    for _, row in df_subset.iterrows():
        formatted_cells = [format_value(row[col], col, precision) for col in df_subset.columns]
        rows_latex.append(" & ".join(formatted_cells) + r" \\")
        
    # Assemble LaTeX code
    lines = []
    has_table_env = bool(caption or label)
    
    if has_table_env:
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        if caption:
            lines.append(f"\\caption{{{caption}}}")
        if label:
            lines.append(f"\\label{{{label}}}")
            
    lines.append(f"\\begin{{tabular}}{{{alignment}}}")
    
    if booktabs:
        lines.append(r"\toprule")
    else:
        lines.append(r"\hline")
        
    lines.append(" & ".join(formatted_headers) + r" \\")
    
    if booktabs:
        lines.append(r"\midrule")
    else:
        lines.append(r"\hline")
        
    for row_line in rows_latex:
        lines.append(row_line)
        
    if booktabs:
        lines.append(r"\bottomrule")
    else:
        lines.append(r"\hline")
        
    lines.append(r"\end{tabular}")
    
    if has_table_env:
        lines.append(r"\end{table}")
        
    table_code = "\n".join(lines)
    
    if standalone:
        doc_lines = [
            r"\documentclass{article}",
            r"\usepackage{booktabs}" if booktabs else "",
            r"\begin{document}",
            table_code,
            r"\end{document}"
        ]
        # Filter out empty strings
        return "\n".join([l for l in doc_lines if l])
        
    return table_code

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert a CSV file into a LaTeX table for Overleaf/LaTeX documents."
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Path to the input CSV file."
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to save the generated LaTeX file. If not specified, prints to stdout."
    )
    parser.add_argument(
        "-c", "--columns",
        help="Comma-separated list of columns to include. "
             "Can rename columns using 'col_name:New Label' format. "
             "Example: 'date:Timestamp,close:Close Price,tic:Ticker'."
    )
    parser.add_argument(
        "--precision", type=int, default=2,
        help="Decimal precision for float formatting (default: 2)."
    )
    parser.add_argument(
        "--no-bold-headers", action="store_true",
        help="Disable bold formatting for column headers."
    )
    parser.add_argument(
        "--no-booktabs", action="store_true",
        help="Disable booktabs rules and use standard hlines instead."
    )
    parser.add_argument(
        "--caption",
        help="Caption for the LaTeX table."
    )
    parser.add_argument(
        "--label",
        help="LaTeX label for the table (e.g. 'tab:my_data')."
    )
    parser.add_argument(
        "--alignment",
        help="Column alignment specifier (e.g. 'lcr'). If not provided, it is auto-detected."
    )
    parser.add_argument(
        "--standalone", action="store_true",
        help="Wrap the table in a standalone compileable LaTeX document."
    )
    
    args = parser.parse_args()
    
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return 1
        
    columns_map = None
    if args.columns:
        columns_map = {}
        parts = args.columns.split(",")
        for part in parts:
            if ":" in part:
                col_name, label = part.split(":", 1)
                columns_map[col_name.strip()] = label.strip()
            else:
                columns_map[part.strip()] = part.strip()
                
    latex_code = generate_latex_table(
        df=df,
        columns_map=columns_map,
        precision=args.precision,
        bold_headers=not args.no_bold_headers,
        booktabs=not args.no_booktabs,
        caption=args.caption,
        label=args.label,
        alignment=args.alignment,
        standalone=args.standalone
    )
    
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(latex_code + "\n")
            print(f"LaTeX table successfully written to {args.output}")
        except Exception as e:
            print(f"Error writing output file: {e}")
            return 1
    else:
        print(latex_code)
        
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

