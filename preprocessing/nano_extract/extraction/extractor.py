"""
nano_extract.extraction.extractor
====================================
Main extraction pipeline: detect dips → refine boundaries → extract
and label segments.

This is the entry point for processing a DataFrame of nanoclean-
processed traces.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it

from nano_extract.core.config import ExtractionConfig
from nano_extract.detection.boundary_refiner import BoundaryRefiner
from nano_extract.detection.dip_detector import DipDetector

logger = logging.getLogger(__name__)


class SegmentExtractor:
    """Extract labeled segments from nanoclean-processed traces.

    Parameters
    ----------
    config : ExtractionConfig, optional
        Extraction parameters.  See :class:`ExtractionConfig`.

    Example
    -------
    >>> from nano_extract import SegmentExtractor, ExtractionConfig
    >>> cfg = ExtractionConfig(n_expected_dips=5)
    >>> extractor = SegmentExtractor(cfg)
    >>> segments_df = extractor.process_dataframe(cleaned_df)
    """

    def __init__(self, config: Optional[ExtractionConfig] = None):
        self.cfg = config or ExtractionConfig()
        self.detector = DipDetector(self.cfg)
        self.refiner = BoundaryRefiner(self.cfg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_single(
        self,
        signal: np.ndarray,
        run: Optional[str] = None,
        channel: Optional[Any] = None,
        trace_id: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Extract segments from a single cleaned signal.

        Parameters
        ----------
        signal : np.ndarray
            Cleaned signal array.
        run, channel, trace_id : optional
            Metadata carried through to the output dicts.

        Returns
        -------
        list of dict
            One dict per segment with keys: run, channel, trace_id,
            segment_index, start, end, signal, length, and optionally aa.
        """
        signal = np.asarray(signal, dtype=float)

        # 1) Detect dips
        dips = self.detector.detect(signal)
        if len(dips) < 2:
            logger.warning(
                "Only %d dips found for trace %s — need at least 2 to form a segment",
                len(dips),
                trace_id,
            )
            return []

        # 2) Refine boundaries and get segment ranges
        boundaries = self.refiner.get_segment_boundaries(signal, dips)

        # 3) Determine peptide labels if available
        peptide = self.cfg.run_to_peptide.get(run, "") if run else ""

        # 4) Build segment records
        segments: List[Dict[str, Any]] = []
        for i, (seg_start, seg_end) in enumerate(boundaries):
            seg_signal = signal[seg_start:seg_end]

            record: Dict[str, Any] = {
                "run": run,
                "channel": channel,
                "trace_id": trace_id,
                "segment_index": i,
                "start": seg_start,
                "end": seg_end,
                "length": len(seg_signal),
                "signal": seg_signal.tolist(),
            }

            # Label with amino acid if peptide mapping is available
            if peptide and i < len(peptide):
                record["aa"] = peptide[i]
            else:
                record["aa"] = None

            # Add dip metadata for the bounding dips
            if i < len(dips) - 1:
                record["left_dip_min"] = float(dips[i].min_value)
                record["right_dip_min"] = float(dips[i + 1].min_value)

            segments.append(record)

        return segments

    def process_dataframe(
        self,
        df: pd.DataFrame,
        signal_column: str = "cleaned",
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Process an entire DataFrame of nanoclean output.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain at minimum a ``cleaned`` (or signal_column)
            column with array-like signal data.  ``run``, ``channel``,
            and ``trace_id`` columns are used if present.
        signal_column : str
            Name of the column containing cleaned signals.
        show_progress : bool
            Show tqdm progress bar.

        Returns
        -------
        pd.DataFrame
            One row per extracted segment.
        """
        all_segments: List[Dict[str, Any]] = []
        skipped = 0

        iterator = df.iterrows()
        if show_progress:
            iterator = tqdm(iterator, total=len(df), desc="Extracting segments", unit="trace")

        for idx, row in iterator:
            signal = row.get(signal_column)
            if signal is None:
                skipped += 1
                continue

            signal = np.asarray(signal, dtype=float)
            if len(signal) < self.cfg.min_dip_width * 2:
                skipped += 1
                continue

            segments = self.extract_single(
                signal=signal,
                run=row.get("run"),
                channel=row.get("channel"),
                trace_id=row.get("trace_id", idx),
            )
            all_segments.extend(segments)

        result = pd.DataFrame(all_segments)

        logger.info(
            "Extracted %d segments from %d traces (%d skipped)",
            len(result),
            len(df),
            skipped,
        )

        return result

    def process_pickle(
        self,
        path: Union[str, Path],
        signal_column: str = "cleaned",
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Load a pickle file and extract segments.

        Parameters
        ----------
        path : str or Path
            Path to a pandas pickle file (nanoclean output).
        signal_column : str
            Column containing cleaned signals.
        show_progress : bool
            Show progress bar.

        Returns
        -------
        pd.DataFrame
        """
        path = Path(path)
        logger.info("Loading %s", path)
        df = pd.read_pickle(path)
        logger.info("Loaded %d traces", len(df))
        return self.process_dataframe(df, signal_column, show_progress)

    def process_and_save(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        fmt: str = "pkl",
        signal_column: str = "cleaned",
        show_progress: bool = True,
    ) -> Tuple[pd.DataFrame, Path]:
        """Load → extract → save in one call.

        Returns
        -------
        (pd.DataFrame, Path)
            The segments DataFrame and the path it was saved to.
        """
        from datetime import datetime

        segments_df = self.process_pickle(input_path, signal_column, show_progress)

        if segments_df.empty:
            logger.warning("No segments extracted")
            return segments_df, Path()

        out_dir = Path(output_path or self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if fmt == "pkl":
            out_file = out_dir / f"segments_{ts}.pkl"
            segments_df.to_pickle(out_file)
        elif fmt == "csv":
            out_file = out_dir / f"segments_{ts}.csv"
            # Stringify signal column for CSV
            df_out = segments_df.copy()
            df_out["signal"] = df_out["signal"].apply(str)
            df_out.to_csv(out_file, index=False)
        elif fmt == "json":
            out_file = out_dir / f"segments_{ts}.json"
            segments_df.to_json(out_file, orient="records", indent=2)
        else:
            raise ValueError(f"Unknown format: {fmt}")

        logger.info("Saved %d segments → %s", len(segments_df), out_file)
        return segments_df, out_file
