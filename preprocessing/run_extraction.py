#!/usr/bin/env python3
"""
run_extraction.py
=================
Run YY boundary detection and segment extraction on nanoclean output.

Usage:
    python run_extraction.py results/phos_data_denoising_tests/cleaned_20260416_213043.pkl
    python run_extraction.py cleaned.pkl -o segments/ --format csv
    python run_extraction.py cleaned.pkl --n-dips 5 --min-dip-width 150 -v
    python run_extraction.py cleaned.pkl --peptide-map peptides.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from nano_extract import SegmentExtractor, ExtractionConfig

logger = logging.getLogger("nano_extract")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Extract YY-bounded segments from nanoclean-processed signals.",
    )
    p.add_argument(
        "input",
        help="Path to a nanoclean output pickle file.",
    )
    p.add_argument(
        "-o", "--output",
        default=None,
        help="Output file path. Defaults to <input_dir>/segments_<timestamp>.pkl",
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
        help='JSON file mapping run IDs to peptide sequences, '
             'e.g. {"20231124_run01_a": "VXXA"}.',
    )
    p.add_argument(
        "--format",
        choices=["pkl", "csv", "json"],
        default="pkl",
        help="Output format (default: pkl).",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Only keep traces with exactly n_dips-1 segments.",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load peptide mapping
    run_to_peptide = {}
    if args.peptide_map:
        pmap = Path(args.peptide_map)
        if pmap.exists():
            with open(pmap) as f:
                run_to_peptide = json.load(f)
            logger.info("Loaded peptide mapping for %d runs", len(run_to_peptide))
        else:
            logger.warning("Peptide map file not found: %s", pmap)

    # Build config
    config = ExtractionConfig(
        n_expected_dips=args.n_dips,
        min_dip_width=args.min_dip_width,
        dip_threshold_percentile=args.threshold_percentile,
        include_flanks=args.include_flanks,
        run_to_peptide=run_to_peptide,
    )

    # Load and process
    input_path = Path(args.input)
    logger.info("Loading %s", input_path)
    df = pd.read_pickle(input_path)
    logger.info("Loaded %d traces", len(df))

    extractor = SegmentExtractor(config)
    segments_df = extractor.process_dataframe(
        df,
        signal_column=args.signal_column,
    )

    if segments_df.empty:
        logger.warning("No segments extracted.")
        sys.exit(1)

    # Strict mode: only keep traces with the expected number of segments
    expected_segs = args.n_dips - 1
    if args.strict:
        segs_per = segments_df.groupby("trace_id").size()
        good_traces = segs_per[segs_per == expected_segs].index
        before = segments_df["trace_id"].nunique()
        segments_df = segments_df[segments_df["trace_id"].isin(good_traces)]
        after = segments_df["trace_id"].nunique()
        logger.info(
            "Strict mode: kept %d / %d traces with exactly %d segments",
            after, before, expected_segs,
        )

    # Determine output path
    if args.output:
        out_path = Path(args.output)
    else:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = input_path.parent / f"segments_{ts}.{args.format}"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save
    if args.format == "pkl":
        segments_df.to_pickle(out_path)
    elif args.format == "csv":
        df_out = segments_df.copy()
        df_out["signal"] = df_out["signal"].apply(str)
        df_out.to_csv(out_path, index=False)
    elif args.format == "json":
        segments_df.to_json(out_path, orient="records", indent=2)

    logger.info("Saved %d segments → %s", len(segments_df), out_path)

    # Summary
    n_traces = segments_df["trace_id"].nunique()
    segs_per = segments_df.groupby("trace_id").size()
    exact = (segs_per == expected_segs).sum()

    print(f"\n{'='*50}")
    print(f"EXTRACTION SUMMARY")
    print(f"{'='*50}")
    print(f"Input:  {len(df)} traces from {input_path.name}")
    print(f"Output: {len(segments_df)} segments from {n_traces} traces")
    print(f"Traces with {expected_segs} segments: {exact} ({exact/len(df)*100:.1f}%)")
    print()
    print("Segments per trace:")
    for n, c in segs_per.value_counts().sort_index().items():
        print(f"  {n} segments: {c} traces")
    print()
    print("Segment lengths by position:")
    for si in sorted(segments_df["segment_index"].unique()):
        sub = segments_df[segments_df["segment_index"] == si]["length"]
        print(f"  seg {si}: n={len(sub)}, median={sub.median():.0f}, "
              f"min={sub.min()}, max={sub.max()}")
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
