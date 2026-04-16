"""
nanoclean.cli
===================
Command-line interface for the signal cleaning pipeline.

Usage::

    # Clean a JSON file (defaults: isolation → tv → cwt_huber)
    signal-cleaner data.json -o cleaned_output/

    # Clean fast5 files with custom passes
    signal-cleaner reads/ --first-pass cwt_huber --second-pass lowpass --no-third-pass

    # Limit to 50 traces for a quick test
    signal-cleaner data.json --max-traces 50
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from nanoclean.core.config import CleanerConfig
from nanoclean.core.trace import TraceData
from nanoclean.io.loader import load_traces
from nanoclean.processing.cleaner import SignalCleaner

logger = logging.getLogger("nanoclean")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="signal-cleaner",
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

    # ---- Config --------------------------------------------------------
    config = CleanerConfig(
        first_pass_method=args.first_pass,
        second_pass_method=args.second_pass,
        third_pass_cwt=not args.no_third_pass,
        output_dir=args.output_dir,
    )
    cleaner = SignalCleaner(config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load ----------------------------------------------------------
    logger.info("Loading traces from %s", args.input)
    raw_traces = load_traces(
        args.input,
        trace_keys=args.trace_keys,
        max_traces=args.max_traces,
    )
    logger.info("Loaded %d traces", len(raw_traces))

    if not raw_traces:
        logger.warning("No traces found — nothing to do.")
        sys.exit(0)

    # ---- Process -------------------------------------------------------
    results = []
    for t in tqdm(raw_traces, desc="Cleaning"):
        trace = TraceData(
            raw_signal=np.asarray(t["signal"], dtype=float),
            metadata={k: v for k, v in t.items() if k != "signal"},
        )
        cleaner.process(trace)
        results.append(trace.to_dict())

    df = pd.DataFrame(results)

    # ---- Save ----------------------------------------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.format == "pkl":
        out_path = out_dir / f"cleaned_{ts}.pkl"
        df.to_pickle(out_path)
    elif args.format == "csv":
        out_path = out_dir / f"cleaned_{ts}.csv"
        df.to_csv(out_path, index=False)
    elif args.format == "json":
        out_path = out_dir / f"cleaned_{ts}.json"
        df.to_json(out_path, orient="records", indent=2)

    logger.info("Saved %d cleaned traces → %s", len(df), out_path)


if __name__ == "__main__":
    main()
