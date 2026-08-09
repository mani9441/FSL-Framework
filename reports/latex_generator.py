"""
LaTeX Table Generator for FSL Research Framework.
Converts evaluation metrics and comparative tables into publication-ready LaTeX markup.
"""

import pandas as pd
from typing import Dict, Any, List
from utilities.logger import get_logger

logger = get_logger("latex_table_generator")


class LaTeXTableGenerator:
    """Generator converting data tables into publication-quality LaTeX markup."""

    @classmethod
    def generate_latex_table(
        cls,
        df: pd.DataFrame,
        caption: str = "Experimental Results Summary",
        label: str = "tab:results_summary",
    ) -> str:
        """
        Converts pandas DataFrame into LaTeX booktabs table syntax.
        """
        if df.empty:
            return "% Empty DataFrame - No LaTeX table generated.\n"

        cols = df.columns.tolist()
        col_spec = "l" + "r" * (len(cols) - 1)

        latex_lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\toprule",
            " & ".join([cls._clean_tex(c) for c in cols]) + " \\\\",
            "\\midrule",
        ]

        has_na = False
        for _, row in df.iterrows():
            formatted_vals = []
            for col in cols:
                val = row[col]
                if pd.isna(val) or val is None or str(val).strip().lower() in ["nan", "none", "null", "n/a"]:
                    formatted_vals.append("---")
                    has_na = True
                elif isinstance(val, float):
                    formatted_vals.append(f"{val:.4f}" if abs(val) < 100 else f"{val:.1f}")
                else:
                    formatted_vals.append(cls._clean_tex(str(val)))
            latex_lines.append(" & ".join(formatted_vals) + " \\\\")

        if has_na:
            latex_lines.append("% N/A or --- indicates an unexecuted condition due to model context limit.")

        latex_lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])

        tex_code = "\n".join(latex_lines) + "\n"
        logger.debug(f"Generated LaTeX Table: '{caption}' ({len(df)} rows)")
        return tex_code

    @staticmethod
    def _clean_tex(text: str) -> str:
        """Escapes special LaTeX characters."""
        clean = text.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")
        return clean.title()
