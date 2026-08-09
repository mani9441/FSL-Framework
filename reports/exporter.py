"""
Report Exporter for FSL Research Framework.
Exports research reports across Markdown, HTML, LaTeX, JSON, and CSV formats.
"""

from pathlib import Path
from typing import Dict, Any, Union
import pandas as pd

from reports.schema import ResearchReportPayload
from reports.markdown_generator import MarkdownReportGenerator
from utilities.helpers import save_json
from utilities.logger import get_logger

logger = get_logger("report_exporter")


class ReportExporter:
    """Exporter writing multi-format research report artifacts to disk."""

    @classmethod
    def export_all(
        cls, payload: ResearchReportPayload, output_dir: Union[str, Path]
    ) -> Dict[str, Path]:
        """
        Saves research report in .md, .html, .tex, .json, and .csv formats.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        md_path = out_path / "research_report.md"
        html_path = out_path / "research_report.html"
        tex_path = out_path / "evaluation_tables.tex"
        json_path = out_path / "report_summary.json"
        csv_path = out_path / "report_summary.csv"

        # 1. Markdown Export
        md_content = MarkdownReportGenerator.generate_markdown(payload)
        md_path.write_text(md_content, encoding="utf-8")

        # 2. HTML Export (styled HTML wrapper)
        html_content = cls._convert_md_to_html(md_content, title=f"Research Report - {payload.experiment_id}")
        html_path.write_text(html_content, encoding="utf-8")

        # 3. LaTeX Tables Export
        tex_content = payload.evaluation_tables_tex if payload.evaluation_tables_tex else "% No LaTeX tables available.\n"
        tex_path.write_text(tex_content, encoding="utf-8")

        # 4. JSON Summary Export
        save_json(payload.to_dict(), json_path)

        # 5. CSV Summary Export
        flat_summary = {
            "experiment_id": payload.experiment_id,
            "timestamp": payload.timestamp,
            **{f"perf_{k}": v for k, v in payload.performance_report.items() if isinstance(v, (int, float))},
            **{f"eff_{k}": v for k, v in payload.statistical_report.get("efficiency", {}).items() if isinstance(v, (int, float))},
            **{f"rel_{k}": v for k, v in payload.statistical_report.get("reliability", {}).items() if isinstance(v, (int, float))},
        }
        df = pd.DataFrame([flat_summary])
        df.to_csv(csv_path, index=False, encoding="utf-8")

        logger.info(f"Exported all research report formats to {out_path}")
        return {
            "md": md_path,
            "html": html_path,
            "tex": tex_path,
            "json": json_path,
            "csv": csv_path,
        }

    @staticmethod
    def _convert_md_to_html(md_text: str, title: str) -> str:
        """Simple HTML template wrapper around Markdown for local viewing."""
        lines = []
        for line in md_text.split("\n"):
            if line.startswith("# "):
                lines.append(f"<h1>{line[2:]}</h1>")
            elif line.startswith("## "):
                lines.append(f"<h2>{line[3:]}</h2>")
            elif line.startswith("### "):
                lines.append(f"<h3>{line[4:]}</h3>")
            elif line.startswith("- "):
                lines.append(f"<li>{line[2:]}</li>")
            elif line.startswith("> "):
                lines.append(f"<blockquote>{line[2:]}</blockquote>")
            elif line.strip() == "---":
                lines.append("<hr>")
            elif line.strip():
                lines.append(f"<p>{line}</p>")

        body = "\n".join(lines)
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; line-height: 1.6; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #1F2937; background-color: #F9FAFB; }}
        h1 {{ color: #111827; border-bottom: 2px solid #E5E7EB; padding-bottom: 8px; }}
        h2 {{ color: #1F2937; margin-top: 32px; }}
        blockquote {{ background: #EFF6FF; border-left: 4px solid #3B82F6; padding: 12px 16px; margin: 16px 0; border-radius: 4px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 14px; background: white; }}
        th, td {{ border: 1px solid #E5E7EB; padding: 10px 14px; text-align: left; }}
        th {{ background-color: #F3F4F6; font-weight: 600; }}
        hr {{ border: None; border-top: 1px solid #E5E7EB; margin: 30px 0; }}
    </style>
</head>
<body>
{body}
</body>
</html>"""
