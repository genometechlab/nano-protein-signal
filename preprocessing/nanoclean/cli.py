"""
nanoclean.cli
===================
Command-line interface for the signal cleaning pipeline.

Usage::

    # Clean a JSON file with 6 workers (defaults: isolation → tv → cwt_huber)
    nanoclean data.json -o cleaned_output/ --workers 6

    # Clean fast5 files with custom passes
    nanoclean reads/ --first-pass cwt_huber --second-pass lowpass --no-third-pass

    # Limit to 50 traces for a quick test (sequential, no multiprocessing)
    nanoclean data.json --max-traces 50 --workers 1
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("nanoclean")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nanoclean",
        description="Multi-pass signal cleaning for nanopore trace data.",
    )
    p.add_argument(
        "input",
        help="Path to a .fast5, .json file, or directory of .fast5 files.",
    )
    p.add_argument(
        "-o", "--output-dir",
        default="./output",
        help="Directory for output files (default: ./output).",
    )
    p.add_argument(
        "--first-pass",
        default="isolation",
        choices=["cwt_huber", "hampel", "ransac", "isolation"],
        help="Spike detection method (default: isolation).",
    )
    p.add_argument(
        "--second-pass",
        default="tv",
        choices=["lowpass", "bilateral", "tv", "kalman", "wavelet", "none"],
        help="Smoothing method (default: tv).",
    )
    p.add_argument(
        "--no-third-pass",
        action="store_true",
        help="Disable the CWT+Huber refinement pass.",
    )
    p.add_argument(
        "-w", "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: cpu_count - 1, max 8). "
             "Use 1 to disable multiprocessing.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Traces per batch (default: auto-sized for load balancing).",
    )
    p.add_argument(
        "--max-traces",
        type=int,
        default=None,
        help="Process at most N traces (for quick testing).",
    )
    p.add_argument(
        "--trace-keys",
        nargs="*",
        default=None,
        help="Only process these specific trace/read IDs.",
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

    from nanoclean.core.config import CleanerConfig
    from nanoclean.processing.batch import BatchCleaner

    # ---- Config --------------------------------------------------------
    config = CleanerConfig(
        first_pass_method=args.first_pass,
        second_pass_method=args.second_pass,
        third_pass_cwt=not args.no_third_pass,
        output_dir=args.output_dir,
    )

    # ---- Process -------------------------------------------------------
    with BatchCleaner(
        config=config,
        n_workers=args.workers,
        batch_size=args.batch_size,
    ) as bc:
        df, out_path = bc.process_file_and_save(
            path=args.input,
            output_dir=args.output_dir,
            fmt=args.format,
            trace_keys=args.trace_keys,
            max_traces=args.max_traces,
        )

    if df.empty:
        logger.warning("No traces processed.")
        sys.exit(0)

    logger.info("Done — %d traces → %s", len(df), out_path)


if __name__ == "__main__":
    main()