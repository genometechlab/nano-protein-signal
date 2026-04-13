"""Data loading utilities."""

import ast
import json
import pickle
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any, Union

import numpy as np
import pandas as pd
import orjson

logger = logging.getLogger(__name__)


class DataLoader:
    """Data loader for various file formats."""

    def __init__(
        self,
        data_source: str,
        data_type: str,
        signal_dict: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        min_signal_length: Optional[int] = None,
        max_signal_length: Optional[int] = None
    ) -> None:
        self.data_source = data_source
        self.data_type = data_type
        self.signal_dict = signal_dict
        self.metadata = metadata
        self.min_signal_length = min_signal_length
        self.max_signal_length = max_signal_length

        self.metadata_traces = None
        if metadata and 'traces' in metadata:
            self.metadata_traces = metadata['traces']
            logger.info(f"Metadata specifies {len(self.metadata_traces)} traces")

    def load_data(self) -> Optional[Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]]:
        if self.data_type == 'csv':
            return self._load_csv()
        elif self.data_type == 'json':
            return self._load_json()
        elif self.data_type == 'pickle':
            return self._load_pickle()
        else:
            raise ValueError(f"Unsupported data type: {self.data_type}")

    def _load_csv(self) -> Optional[Union[pd.DataFrame, List[Dict[str, Any]]]]:
        try:
            data = pd.read_csv(self.data_source)
            if self.signal_dict:
                return self._convert_to_signal_dict(data, apply_length_filter=True)
            return data
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            return None

    def _load_json(self) -> Optional[Dict[str, Any]]:
        try:
            with open(self.data_source, 'rb') as file:
                data = orjson.loads(file.read())
            return data
        except Exception as e:
            logger.error(f"Error loading JSON file: {e}")
            return None

    def _load_pickle(self) -> Optional[Union[pd.DataFrame, List[Dict[str, Any]]]]:
        try:
            with open(self.data_source, 'rb') as f:
                data = pickle.load(f)

            if isinstance(data, list):
                dataframe = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                dataframe = data
            elif isinstance(data, dict) and 'data' in data:
                dataframe = data['data']
            else:
                raise ValueError(f"Unexpected pickle data type: {type(data)}")

            if self.signal_dict:
                return self._convert_to_signal_dict(dataframe, apply_length_filter=False)

            return dataframe
        except Exception as e:
            logger.error(f"Error loading pickle file: {e}")
            return None

    def _convert_to_signal_dict(
        self,
        df: pd.DataFrame,
        apply_length_filter: bool = True
    ) -> List[Dict[str, Any]]:
        """Convert dataframe to signal dictionary format."""
        aa_field = self._find_aa_field(df)
        segment_field = self._find_segment_field(df)

        if self.metadata_traces:
            df = self._filter_by_metadata(df)
            logger.info(f"After metadata filtering: {len(df)} rows remain")

        if apply_length_filter and (self.min_signal_length or self.max_signal_length):
            df = self._filter_by_length(df, segment_field)
            logger.debug(f"After length filter: {len(df)} records")

        records = []
        for _, row in df.iterrows():
            record = {
                'run': row.get('run', ''),
                'channel': row.get('channel', 0),
                'segment': row.get('df_index', row.get('segment', 0)),
                'aa': row.get(aa_field, '') if aa_field else '',
                'cleaned_segment': row.get(segment_field, []) if segment_field else [],
                'pretty': row.get(aa_field, '') if aa_field else ''
            }
            records.append(record)

        return records

    def _find_aa_field(self, df: pd.DataFrame) -> Optional[str]:
        """Find the amino acid field in dataframe."""
        candidates = ['variable_region', 'aa', 'label']
        for field in candidates:
            if field in df.columns:
                unique_vals = df[field].unique()
                if any(str(v) in 'ACDEFGHIKLMNPQRSTVWY' for v in unique_vals):
                    return field
        return None

    def _find_segment_field(self, df: pd.DataFrame) -> Optional[str]:
        """Find the segment data field in dataframe."""
        candidates = ['cleaned_segments', 'cleaned_segment', 'raw_segments', 'segments']
        for field in candidates:
            if field in df.columns:
                return field
        return None

    def _filter_by_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter dataframe by metadata traces."""
        required_cols = {'channel', 'variable_region', 'metadata'}
        missing = required_cols - set(df.columns)
        if missing:
            logger.warning(f"Cannot apply metadata filter, missing columns: {missing}")
            return df

        filtered_dfs = []

        for trace in self.metadata_traces:
            channel = float(trace['Channel'])
            amino_acid = trace['AA']

            conditions = (
                (df['channel'] == channel) &
                (df['variable_region'] == amino_acid)
            )

            if 'full_metadata' in trace and trace['full_metadata']:
                conditions = conditions & (df['metadata'] == trace['full_metadata'])
            else:
                peptide_group = trace['Run']
                conditions = conditions & (df['metadata'].str.startswith(peptide_group))

            if 'df_index' in trace and trace['df_index'] is not None:
                conditions = conditions & (df['df_index'].astype(str) == str(trace['df_index']))

            trace_data = df[conditions]

            if trace_data.empty:
                trace_id = trace.get('full_metadata', trace['Run'])
                logger.warning(f"No match for {trace_id} Ch{channel} AA={amino_acid} idx={trace.get('df_index')}")
                continue

            if len(trace_data) > 1:
                logger.warning(f"Multiple matches ({len(trace_data)}) for trace, taking first")
                trace_data = trace_data.head(1)

            filtered_dfs.append(trace_data)

        if not filtered_dfs:
            logger.warning("No traces matched metadata filters")
            return pd.DataFrame()

        result = pd.concat(filtered_dfs, ignore_index=True)
        result = result.drop_duplicates(subset=['metadata', 'channel', 'variable_region', 'df_index'])
        logger.info(f"Metadata filtering: {len(result)} unique traces matched from {len(self.metadata_traces)} specified")
        return result

    def _filter_by_length(self, df: pd.DataFrame, segment_field: str) -> pd.DataFrame:
        """Filter dataframe by signal length."""
        if not segment_field:
            return df

        df = df.copy()
        df['_signal_length'] = df[segment_field].apply(_get_signal_length)

        if self.min_signal_length:
            df = df[df['_signal_length'] >= self.min_signal_length]
        if self.max_signal_length:
            df = df[df['_signal_length'] <= self.max_signal_length]

        return df.drop('_signal_length', axis=1)


def _get_signal_length(signal_value: Any) -> int:
    """Get the length of a signal value in various storage formats."""
    try:
        if isinstance(signal_value, (list, np.ndarray)):
            return len(signal_value)
        if isinstance(signal_value, str):
            return len(_parse_string_signal(signal_value))
        return 0
    except Exception:
        return 0


def _parse_string_signal(value: str) -> list:
    """Parse a string-encoded signal (Python literal, JSON, or CSV)."""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        pass

    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        pass

    return value.split(',')


def parse_signal_data(signal_value: Any) -> np.ndarray:
    """Parse signal data from various formats to numpy array."""
    if isinstance(signal_value, np.ndarray):
        return signal_value.astype(np.float64)
    if isinstance(signal_value, list):
        return np.array(signal_value, dtype=np.float64)
    if isinstance(signal_value, str):
        parsed = _parse_string_signal(signal_value)
        return np.array([float(x) if isinstance(x, str) else x for x in parsed],
                        dtype=np.float64)
    raise ValueError(f"Cannot parse signal data of type {type(signal_value)}")


def process_pre_segmented_data(raw_segments: List[Any]) -> Dict[str, Any]:
    """Convert pre-segmented data to the format expected by the HMM."""
    if not isinstance(raw_segments, list):
        raw_segments = [raw_segments]

    means = []
    variances = []
    breakpoints = [0]
    current_pos = 0

    for segment in raw_segments:
        if isinstance(segment, (list, np.ndarray)):
            segment_array = np.array(segment).flatten()
        else:
            segment_array = np.array([segment])

        means.append(float(np.mean(segment_array)))
        variances.append(float(np.var(segment_array)))

        current_pos += len(segment_array)
        breakpoints.append(current_pos)

    return {
        'means': np.array(means),
        'variances': np.array(variances),
        'breakpoints': breakpoints
    }