"""
nano_extract.cli
=================
Command-line interface for YY boundary detection and segment extraction.

Usage::

    # Extract segments from nanoclean output
    nano-extract cleaned.pkl -o segments/

    # With peptide labeling
    nano-extract cleaned.pkl --peptide-map peptides.json

    # Custom dip parameters
    nano-extract cleaned.pkl --n-dips 5 --min-dip-width 200
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("nano_extract")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nano-extract",
        description="Extract YY-bounded segments from nanoclean-processed signals.",
    )
    p.add_argument(
        "input",
        help="Path to a nanoclean output pickle file.",
    )
    p.add_argument(
        "-o", "--output-dir",
        default="./extracted",
        help="Directory for output files (default: ./extracted).",
    )
    p.add_argument(
        "--signal-column",
        default="cleaned",
        help="DataFrame column containing signal data (default: cleaned).",
    )
    p.add_argument(
        "--n-dips",
        type=int,
        default=5,
        help="Expected number of YY dips per trace (default: 5).",
    )
    p.add_argument(
        "--min-dip-width",
        type=int,
        default=200,
        help="Minimum sustained dip width in samples (default: 200).",
    )
    p.add_argument(
        "--threshold-percentile",
        type=float,
        default=30.0,
        help="Percentile for dip threshold (default: 30).",
    )
    p.add_argument(
        "--include-flanks",
        action="store_true",
        help="Also extract flank regions outside the outermost dips.",
    )
    p.add_argument(
        "--peptide-map",
        default=None,
        help="JSON file mapping run IDs to peptide sequences, "
             'e.g. {"20231124_run01_a": "HDKER"}.',
    )
    p.add_argument(
        "--format",
        choices=["pkl", "csv", "json"],
        default="pkl",
        help="Output format (default: pkl).",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    from nano_extract.core.config import ExtractionConfig
    from nano_extract.extraction.extractor import SegmentExtractor

    # Load peptide mapping if provided
    run_to_peptide = {}
    if args.peptide_map:
        pmap = Path(args.peptide_map)
        if pmap.exists():
            with open(pmap) as f:
                run_to_peptide = json.load(f)
            logger.info("Loaded peptide mapping for %d runs", len(run_to_peptide))
        else:
            logger.warning("Peptide map file not found: %s", pmap)

    config = ExtractionConfig(
        n_expected_dips=args.n_dips,
        min_dip_width=args.min_dip_width,
        dip_threshold_percentile=args.threshold_percentile,
        include_flanks=args.include_flanks,
        run_to_peptide=run_to_peptide,
        output_dir=args.output_dir,
    )

    extractor = SegmentExtractor(config)
    df, out_path = extractor.process_and_save(
        input_path=args.input,
        output_path=args.output_dir,
        fmt=args.format,
        signal_column=args.signal_column,
    )

    if df.empty:
        logger.warning("No segments extracted.")
        sys.exit(0)

    # Summary
    logger.info("Done — %d segments from %d traces", len(df), df["trace_id"].nunique())
    logger.info("Segments per trace: %.1f avg", len(df) / df["trace_id"].nunique())
    logger.info("Segment length: min=%d, max=%d, median=%.0f",
                df["length"].min(), df["length"].max(), df["length"].median())


if __name__ == "__main__":
    main()
