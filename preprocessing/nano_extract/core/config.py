"""
nano_extract.core.config
=========================
Single source of truth for all extraction parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ExtractionConfig:
    """Complete configuration for the extraction pipeline.

    Defaults are calibrated against nanoclean-processed signals
    sampled at 3012 Hz with the symmetric SSGGYYGGSS flanking structure.

    The construct has 5 YY dip positions and 4 segments between them::

        C- ... YY ... [seg0] ... YY ... [seg1] ... YY ... [seg2] ... YY ... [seg3] ... YY -N
    """

    # ------------------------------------------------------------------
    # Smoothing (applied before dip detection, NOT signal cleaning)
    # ------------------------------------------------------------------
    smoothing_window: int = 101
    """Savitzky-Golay window size for pre-detection smoothing."""

    smoothing_polyorder: int = 3
    """Savitzky-Golay polynomial order."""

    # ------------------------------------------------------------------
    # YY dip detection
    # ------------------------------------------------------------------
    n_expected_dips: int = 5
    """Expected number of YY dip regions per trace.  5 dips yield
    4 inter-dip segments matching the construct structure."""

    dip_threshold_percentile: float = 30.0
    """Percentile of the smoothed signal used as the low-current threshold.
    Points below this are considered 'in a dip'."""

    min_dip_width: int = 200
    """Minimum number of contiguous below-threshold samples for a region
    to qualify as a sustained YY dip."""

    # ------------------------------------------------------------------
    # Boundary refinement
    # ------------------------------------------------------------------
    refinement_padding: int = 50
    """Samples to search around each dip edge when refining boundaries."""

    # ------------------------------------------------------------------
    # Segment extraction
    # ------------------------------------------------------------------
    include_flanks: bool = False
    """If True, also extract the regions before the first dip and after
    the last dip as flank segments."""

    min_segment_length: int = 50
    """Discard segments shorter than this (likely artefacts).
    Set low because single-AA segments (V, A) are legitimately short."""

    # ------------------------------------------------------------------
    # Peptide labelling (optional)
    # ------------------------------------------------------------------
    run_to_peptide: Dict[str, str] = field(default_factory=dict)
    """Mapping of run ID → peptide sequence string (e.g. 'VXXA').
    When provided, segments are labelled with their amino acid.
    Must have exactly n_expected_dips - 1 characters."""

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    output_dir: str = "./extracted"
