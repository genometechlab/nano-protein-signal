"""
signal_cleaner.io.loader
=========================
Unified data loading and saving for ``.fast5`` (HDF5) and ``.json`` trace
files, plus result persistence utilities.

Both loaders return the same format — a list of plain dicts — so the rest
of the pipeline never needs to know which file type was used::

    [
        {
            "trace_id": "read_001",
            "signal": np.array([...]),
            "channel": 1,
            "run": "run_abc",
            ...                       # any extra metadata from the file
        },
        ...
    ]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =====================================================================
# Public loading API
# =====================================================================


def load_traces(
    path: Union[str, Path],
    trace_keys: Optional[List[str]] = None,
    max_traces: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Auto-detect file type and load traces.

    Parameters
    ----------
    path : str or Path
        Path to a ``.fast5``, ``.json``, or directory of ``.fast5`` files.
    trace_keys : list of str, optional
        If given, only load traces whose id / read_id matches one of these.
    max_traces : int, optional
        Cap the number of traces returned (handy for quick tests).

    Returns
    -------
    list of dict
        Each dict has at minimum ``"trace_id"`` and ``"signal"`` keys.
    """
    path = Path(path)

    if path.is_dir():
        traces: List[Dict[str, Any]] = []
        for f5 in sorted(path.rglob("*.fast5")):
            traces.extend(load_fast5(f5, trace_keys=trace_keys))
            if max_traces and len(traces) >= max_traces:
                break
        return traces[:max_traces] if max_traces else traces

    suffix = path.suffix.lower()
    if suffix in (".fast5", ".h5", ".hdf5"):
        traces = load_fast5(path, trace_keys=trace_keys)
    elif suffix == ".json":
        traces = load_json(path, trace_keys=trace_keys)
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Use .fast5, .h5, .hdf5, or .json"
        )

    return traces[:max_traces] if max_traces else traces


# =====================================================================
# .fast5 / HDF5 loader
# =====================================================================


def load_fast5(
    path: Union[str, Path],
    trace_keys: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Load signal traces from an Oxford Nanopore ``.fast5`` file.

    Supports both **single-read** and **multi-read** fast5 layouts:

    Multi-read layout (most common)::

        /read_<uuid>/Raw/Signal

    Single-read layout::

        /Raw/Reads/Read_<n>/Signal

    Channel and tracking metadata are extracted when present.
    """
    import h5py

    path = Path(path)
    traces: List[Dict[str, Any]] = []
    key_set = set(trace_keys) if trace_keys else None

    with h5py.File(path, "r") as f5:
        # --- Multi-read fast5 -------------------------------------------
        read_groups = [k for k in f5.keys() if k.startswith("read_")]
        if read_groups:
            for rg in read_groups:
                read_id = rg.replace("read_", "")
                if key_set and read_id not in key_set:
                    continue
                trace = _extract_multi_read(f5[rg], read_id, path.name)
                if trace is not None:
                    traces.append(trace)
            return traces

        # --- Single-read fast5 ------------------------------------------
        if "Raw" in f5 and "Reads" in f5["Raw"]:
            for read_name in f5["Raw/Reads"]:
                grp = f5[f"Raw/Reads/{read_name}"]
                read_id = str(grp.attrs.get("read_id", read_name))
                if key_set and read_id not in key_set:
                    continue
                trace = _extract_single_read(f5, grp, read_id, path.name)
                if trace is not None:
                    traces.append(trace)
            return traces

        # --- Flat signal key (rare) -------------------------------------
        if "Signal" in f5:
            signal = np.array(f5["Signal"], dtype=float)
            traces.append({
                "trace_id": path.stem,
                "signal": signal,
                "source_file": path.name,
            })
            return traces

    logger.warning("No recognisable signal data in %s", path)
    return traces


def _extract_multi_read(
    read_grp, read_id: str, filename: str
) -> Optional[Dict[str, Any]]:
    """Pull signal + metadata from a multi-read group."""
    signal_path = None
    for candidate in ("Raw/Signal", "Raw/signal", "Signal"):
        if candidate in read_grp:
            signal_path = candidate
            break

    if signal_path is None:
        logger.debug("Skipping read %s — no signal dataset found", read_id)
        return None

    signal = np.array(read_grp[signal_path], dtype=float)
    meta: Dict[str, Any] = {
        "trace_id": read_id,
        "signal": signal,
        "source_file": filename,
    }

    # Channel info
    if "channel_id" in read_grp:
        ch = read_grp["channel_id"]
        meta["channel"] = _safe_attr(ch, "channel_number")
        meta["sampling_rate"] = _safe_attr(ch, "sampling_rate", float)
        meta["digitisation"] = _safe_attr(ch, "digitisation", float)
        meta["offset"] = _safe_attr(ch, "offset", float)
        meta["range"] = _safe_attr(ch, "range", float)

    # Tracking / context
    if "tracking_id" in read_grp:
        trk = read_grp["tracking_id"]
        meta["run"] = _safe_attr(trk, "run_id")
        meta["device_id"] = _safe_attr(trk, "device_id")
        meta["exp_start_time"] = _safe_attr(trk, "exp_start_time")

    # Read-level attrs
    raw_grp = read_grp.get("Raw")
    if raw_grp is not None:
        meta["read_number"] = _safe_attr(raw_grp, "read_number", int)
        meta["start_time"] = _safe_attr(raw_grp, "start_time", int)
        meta["duration"] = _safe_attr(raw_grp, "duration", int)

    return meta


def _extract_single_read(f5, read_grp, read_id: str, filename: str):
    """Pull signal + metadata from a single-read fast5."""
    if "Signal" not in read_grp:
        return None

    signal = np.array(read_grp["Signal"], dtype=float)
    meta: Dict[str, Any] = {
        "trace_id": read_id,
        "signal": signal,
        "source_file": filename,
    }

    if "UniqueGlobalKey/channel_id" in f5:
        ch = f5["UniqueGlobalKey/channel_id"]
        meta["channel"] = _safe_attr(ch, "channel_number")
        meta["sampling_rate"] = _safe_attr(ch, "sampling_rate", float)

    if "UniqueGlobalKey/tracking_id" in f5:
        trk = f5["UniqueGlobalKey/tracking_id"]
        meta["run"] = _safe_attr(trk, "run_id")

    return meta


def _safe_attr(grp, key: str, cast=None):
    """Read an HDF5 attribute, returning None on missing / decode errors."""
    try:
        val = grp.attrs[key]
        if isinstance(val, bytes):
            val = val.decode()
        return cast(val) if cast else val
    except (KeyError, ValueError, UnicodeDecodeError):
        return None


# =====================================================================
# .json loader
# =====================================================================


def load_json(
    path: Union[str, Path],
    trace_keys: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Load traces from a JSON file.

    Supports four layouts:

    1. **List of trace objects**::

        [{"trace_id": "t1", "raw": [...], "aa": "A"}, ...]

    2. **Dict with a ``traces`` key**::

        {"traces": [<same as above>]}

    3. **Column-oriented dict** (keys are field names, values are
       dicts mapping trace_id → value)::

        {"raw": {"t1": [...], "t2": [...]}, "aa": {"t1": "A", "t2": "G"}}

    4. **Single trace dict** (fallback — any dict with a ``raw`` or
       ``signal`` key that doesn't match the above)::

        {"raw": [...], "aa": "A"}
    """
    import orjson

    path = Path(path)
    with open(path, "rb") as fh:
        data = orjson.loads(fh.read())

    key_set = set(trace_keys) if trace_keys else None
    raw_traces = _normalize_json(data)

    traces: List[Dict[str, Any]] = []
    for t in raw_traces:
        tid = t.get("trace_id", t.get("id"))
        if key_set and tid is not None and tid not in key_set:
            continue

        signal = t.get("raw", t.get("signal"))
        if signal is None:
            logger.debug("Skipping trace %s — no signal data", tid)
            continue

        entry: Dict[str, Any] = {
            "trace_id": tid,
            "signal": np.array(signal, dtype=float),
            "source_file": path.name,
        }

        # Carry forward all known optional fields (matches original io.py)
        for field in ("aa", "channel", "run", "metadata"):
            if field in t:
                entry[field] = t[field]

        traces.append(entry)

    logger.info("Loaded %d traces from %s", len(traces), path)
    return traces


def _normalize_json(data) -> List[Dict[str, Any]]:
    """Convert any of the four supported JSON shapes into a flat list."""
    # Shape 1 — already a list
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        # Shape 2 — wrapper with a "traces" key
        if "traces" in data:
            return data["traces"]

        # Shape 3 — column-oriented (raw is a dict of trace_id → signal)
        if "raw" in data and isinstance(data["raw"], dict):
            trace_ids = list(data["raw"].keys())
            traces = []
            for tid in trace_ids:
                trace: Dict[str, Any] = {
                    "trace_id": tid,
                    "raw": data["raw"][tid],
                }
                # Pick up every other top-level field that has this trace_id
                for field in data:
                    if field == "raw":
                        continue
                    if isinstance(data[field], dict) and tid in data[field]:
                        trace[field] = data[field][tid]
                traces.append(trace)
            return traces

        # Shape 4 — single trace dict (fallback)
        return [data]

    return []


# =====================================================================
# Result saving / loading
# =====================================================================


def save_results(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    fmt: str = "pickle",
) -> None:
    """Save processing results to disk.

    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe.
    output_path : str or Path
        Destination file path.
    fmt : str
        One of ``'pickle'``, ``'csv'``, ``'json'``, ``'parquet'``.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "pickle":
        df.to_pickle(output_path)
    elif fmt == "csv":
        df_csv = df.copy()
        # Stringify array/list columns so CSV doesn't choke
        for col in ("raw", "cleaned", "segments", "breakpoints"):
            if col in df_csv.columns:
                df_csv[col] = df_csv[col].apply(str)
        df_csv.to_csv(output_path, index=False)
    elif fmt == "json":
        df.to_json(output_path, orient="records", indent=2)
    elif fmt == "parquet":
        df.to_parquet(output_path)
    else:
        raise ValueError(f"Unknown format: {fmt}")

    logger.info("Saved %d results to %s", len(df), output_path)


def load_results(file_path: Union[str, Path]) -> pd.DataFrame:
    """Reload a previously saved results file.

    Auto-detects format from the file extension.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix in (".pkl", ".pickle"):
        return pd.read_pickle(file_path)
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix == ".json":
        return pd.read_json(file_path)
    if suffix == ".parquet":
        return pd.read_parquet(file_path)

    # Fall back to pickle
    return pd.read_pickle(file_path)


# =====================================================================
# Validation
# =====================================================================


def validate_trace_data(trace_dict: Dict[str, Any]) -> bool:
    """Check that a trace dict has usable signal data.

    Returns ``True`` if the dict contains a ``raw`` or ``signal`` key whose
    value is array-like with length >= 10.
    """
    signal = trace_dict.get("raw", trace_dict.get("signal"))
    if signal is None:
        return False
    try:
        if not hasattr(signal, "__len__") or len(signal) < 10:
            return False
    except Exception:
        return False
    return True


# =====================================================================
# Batch file processing
# =====================================================================


def batch_process_files(
    file_paths: List[Union[str, Path]],
    pipeline_func: Callable,
    n_workers: int = 4,
) -> pd.DataFrame:
    """Process multiple files in parallel and concatenate results.

    Parameters
    ----------
    file_paths : list of str or Path
        Files to process.
    pipeline_func : callable
        Function that takes a file path and returns a DataFrame.
    n_workers : int
        Number of parallel workers.

    Returns
    -------
    pd.DataFrame
        Combined results from all files.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from tqdm import tqdm

    all_results: List[pd.DataFrame] = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_file = {
            executor.submit(pipeline_func, fp): fp for fp in file_paths
        }

        for future in tqdm(
            as_completed(future_to_file),
            total=len(file_paths),
            desc="Processing files",
        ):
            fp = future_to_file[future]
            try:
                result = future.result()
                result["source_file"] = str(fp)
                all_results.append(result)
            except Exception as e:
                logger.error("Error processing %s: %s", fp, e)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        logger.info(
            "Processed %d total traces from %d files",
            len(combined),
            len(file_paths),
        )
        return combined

    return pd.DataFrame()
