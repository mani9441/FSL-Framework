"""
Report Generation Package for FSL Research Framework.
Provides research report payload schema, LaTeX table generator, Markdown report generator, multi-format exporter, and master ReportGenerationEngine.
"""

from reports.schema import ResearchReportPayload
from reports.latex_generator import LaTeXTableGenerator
from reports.markdown_generator import MarkdownReportGenerator
from reports.exporter import ReportExporter
from reports.engine import ReportGenerationEngine

__all__ = [
    "ResearchReportPayload",
    "LaTeXTableGenerator",
    "MarkdownReportGenerator",
    "ReportExporter",
    "ReportGenerationEngine",
]
