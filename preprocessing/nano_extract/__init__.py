"""
nano_extract
=============
YY boundary detection and segment extraction for nanopore peptide signals.

Works on nanoclean-processed signals to find sustained YY dip regions,
refine their boundaries, and extract labeled segments.

Quick start::

    from nano_extract import SegmentExtractor
    segments_df = SegmentExtractor().process_pickle("cleaned.pkl")

With peptide labeling::

    from nano_extract import SegmentExtractor, ExtractionConfig
    cfg = ExtractionConfig(
        n_expected_dips=5,
        run_to_peptide={"20231124_run01_a": "HDKER"},
    )
    segments_df = SegmentExtractor(cfg).process_pickle("cleaned.pkl")
"""

__version__ = "0.1.0"

from nano_extract.core.config import ExtractionConfig
from nano_extract.detection.dip_detector import DipDetector, DipRegion
from nano_extract.detection.boundary_refiner import BoundaryRefiner
from nano_extract.extraction.extractor import SegmentExtractor

__all__ = [
    "ExtractionConfig",
    "DipDetector",
    "DipRegion",
    "BoundaryRefiner",
    "SegmentExtractor",
]
