"""
Statistical Analysis Package for FSL Research Framework.
Provides MetricsAggregator, FactorComparator, SignificanceTester, StatisticalAnalysisEngine, StatisticalAnalysisResult, and DissertationArtifactGenerator.
"""

from analysis.aggregator import MetricsAggregator
from analysis.comparator import FactorComparator
from analysis.significance import SignificanceTester
from analysis.engine import StatisticalAnalysisEngine, StatisticalAnalysisResult
from analysis.dissertation_generator import DissertationArtifactGenerator

__all__ = [
    "MetricsAggregator",
    "FactorComparator",
    "SignificanceTester",
    "StatisticalAnalysisEngine",
    "StatisticalAnalysisResult",
    "DissertationArtifactGenerator",
]
