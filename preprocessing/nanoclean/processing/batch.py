"""
nanoclean.processing.batch
============================
Multiprocessing batch cleaner.

Splits a list of traces into batches, fans them out across a worker pool,
and collects results into a DataFrame — all with a progress bar.

Usage::

    from nanoclean.processing.batch import BatchCleaner

    bc = BatchCleaner(config, n_workers=6)
    df = bc.process_traces(traces)      # list of dicts from load_traces()
    df = bc.process_file("data.json")   # or pass a file path directly
    bc.shutdown()                        # clean up the pool when done
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from nanoclean.core.config import CleanerConfig
from nanoclean.core.trace import TraceData

logger = logging.getLogger(__name__)


# =====================================================================
# Worker functions (module-level so they're picklable)
# =====================================================================

# Each worker process stores its own SignalCleaner instance here after
# initialisation so we don't re-import heavy modules for every batch.
_worker_cleaner = None


def _worker_init(config: CleanerConfig) -> None:
    """Called once per worker process — builds a SignalCleaner."""
    global _worker_cleaner
    from nanoclean.processing.cleaner import SignalCleaner

    _worker_cleaner = SignalCleaner(config)
    logger.debug("Worker %s initialised", mp.current_process().name)


def _process_batch(
    batch: List[Dict[str, Any]],
) -> List[Optional[Dict[str, Any]]]:
    """Process a batch of trace dicts in a single worker.

    Returns a list of result dicts (or None for failed traces).
    """
    global _worker_cleaner
    results: List[Optional[Dict[str, Any]]] = []

    for t in batch:
        try:
            signal = t.get("signal", t.get("raw"))
            if signal is None:
                results.append(None)
                continue

            trace = TraceData(
                raw_signal=np.asarray(signal, dtype=float),
                metadata={k: v for k, v in t.items() if k not in ("signal", "raw")},
            )
            _worker_cleaner.process(trace)
            results.append(trace.to_dict())

        except Exception as e:
            logger.error("Error processing trace in batch: %s", e)
            results.append(None)

    return results


# =====================================================================
# Public API
# =====================================================================


class BatchCleaner:
    """Multiprocessing batch cleaner with a persistent worker pool.

    Parameters
    ----------
    config : CleanerConfig, optional
        Pipeline configuration (defaults used if omitted).
    n_workers : int, optional
        Number of worker processes.  Defaults to ``min(cpu_count - 1, 8)``.
    batch_size : int, optional
        Traces per batch.  If ``None`` (default), automatically sized to
        create ~4 batches per worker for good load balancing.
    """

    def __init__(
        self,
        config: Optional[CleanerConfig] = None,
        n_workers: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        self.config = config or CleanerConfig()
        self.n_workers = n_workers or min(mp.cpu_count() - 1, 8)
        self.batch_size = batch_size
        self._pool: Optional[Pool] = None

    # ------------------------------------------------------------------
    # Pool management
    # ------------------------------------------------------------------

    def _get_pool(self) -> Pool:
        """Lazily create the worker pool on first use."""
        if self._pool is None:
            self._pool = Pool(
                processes=self.n_workers,
                initializer=_worker_init,
                initargs=(self.config,),
            )
            logger.info("Created worker pool with %d workers", self.n_workers)
        return self._pool

    def shutdown(self) -> None:
        """Shut down the worker pool.  Safe to call multiple times."""
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
            logger.info("Worker pool shut down")

    def __del__(self):
        self.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process_traces(
        self,
        traces: List[Dict[str, Any]],
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Clean a list of trace dicts using the worker pool.

        Parameters
        ----------
        traces : list of dict
            Each dict must have a ``"signal"`` (or ``"raw"``) key with an
            array-like value.  Typically the output of ``load_traces()``.
        show_progress : bool
            Show a tqdm progress bar (default True).

        Returns
        -------
        pd.DataFrame
            One row per successfully cleaned trace.
        """
        n = len(traces)
        if n == 0:
            return pd.DataFrame()

        # For tiny datasets, skip the pool overhead
        if n < 4:
            logger.info("Small dataset (%d traces) — processing sequentially", n)
            return self._process_sequential(traces, show_progress)

        # Calculate batch size: aim for ~4 batches per worker
        bs = self.batch_size or max(1, n // (self.n_workers * 4))
        batches = [traces[i : i + bs] for i in range(0, n, bs)]

        logger.info(
            "Processing %d traces in %d batches across %d workers",
            n,
            len(batches),
            self.n_workers,
        )

        pool = self._get_pool()

        try:
            iterator = pool.imap(_process_batch, batches)
            if show_progress:
                iterator = tqdm(
                    iterator,
                    total=len(batches),
                    desc="Cleaning batches",
                    unit="batch",
                )

            results = [
                row
                for batch_result in iterator
                for row in batch_result
                if row is not None
            ]

        except Exception as e:
            logger.error("Multiprocessing failed, falling back to sequential: %s", e)
            self.shutdown()
            return self._process_sequential(traces, show_progress)

        df = pd.DataFrame(results)
        logger.info("Cleaned %d / %d traces successfully", len(df), n)
        return df

    def process_file(
        self,
        path: Union[str, Path],
        trace_keys: Optional[List[str]] = None,
        max_traces: Optional[int] = None,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Load a file and clean all traces using the worker pool.

        Parameters
        ----------
        path : str or Path
            A ``.fast5``, ``.json``, or directory of ``.fast5`` files.
        trace_keys : list of str, optional
            Only process these trace / read IDs.
        max_traces : int, optional
            Cap the number of traces loaded.
        show_progress : bool
            Show progress bar.

        Returns
        -------
        pd.DataFrame
            Cleaned results.
        """
        from nanoclean.io.loader import load_traces

        logger.info("Loading traces from %s", path)
        traces = load_traces(path, trace_keys=trace_keys, max_traces=max_traces)
        logger.info("Loaded %d traces", len(traces))

        if not traces:
            logger.warning("No traces found in %s", path)
            return pd.DataFrame()

        return self.process_traces(traces, show_progress=show_progress)

    def process_file_and_save(
        self,
        path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        fmt: str = "pkl",
        trace_keys: Optional[List[str]] = None,
        max_traces: Optional[int] = None,
        show_progress: bool = True,
    ) -> Tuple[pd.DataFrame, Path]:
        """Load → clean → save in one call.

        Returns the DataFrame and the path it was saved to.
        """
        df = self.process_file(
            path,
            trace_keys=trace_keys,
            max_traces=max_traces,
            show_progress=show_progress,
        )

        if df.empty:
            return df, Path()

        out_dir = Path(output_dir or self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if fmt == "pkl":
            out_path = out_dir / f"cleaned_{ts}.pkl"
            df.to_pickle(out_path)
        elif fmt == "csv":
            out_path = out_dir / f"cleaned_{ts}.csv"
            df.to_csv(out_path, index=False)
        elif fmt == "json":
            out_path = out_dir / f"cleaned_{ts}.json"
            df.to_json(out_path, orient="records", indent=2)
        else:
            raise ValueError(f"Unknown format: {fmt}")

        logger.info("Saved %d cleaned traces → %s", len(df), out_path)
        return df, out_path

    # ------------------------------------------------------------------
    # Sequential fallback
    # ------------------------------------------------------------------

    def _process_sequential(
        self,
        traces: List[Dict[str, Any]],
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Process traces one at a time (fallback / small datasets)."""
        from nanoclean.processing.cleaner import SignalCleaner

        cleaner = SignalCleaner(self.config)
        results: List[Dict[str, Any]] = []
        it = tqdm(traces, desc="Cleaning", unit="trace") if show_progress else traces

        for t in it:
            try:
                signal = t.get("signal", t.get("raw"))
                if signal is None:
                    continue

                trace = TraceData(
                    raw_signal=np.asarray(signal, dtype=float),
                    metadata={k: v for k, v in t.items() if k not in ("signal", "raw")},
                )
                cleaner.process(trace)
                results.append(trace.to_dict())

            except Exception as e:
                logger.error("Error processing trace: %s", e)

        return pd.DataFrame(results)