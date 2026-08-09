"""
Dataset Management Package for FSL Research Framework.
Provides dataset loading, cleaning, record formatting, sampling, and pipeline management.
"""

from datasets.schema import StandardRecord, DatasetMetadata
from datasets.loader import DatasetLoader
from datasets.processor import DatasetProcessor
from datasets.formatter import DatasetFormatter
from datasets.sampler import DatasetSampler
from datasets.manager import DatasetManager

__all__ = [
    "StandardRecord",
    "DatasetMetadata",
    "DatasetLoader",
    "DatasetProcessor",
    "DatasetFormatter",
    "DatasetSampler",
    "DatasetManager",
]
