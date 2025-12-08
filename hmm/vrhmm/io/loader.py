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

        logger.info(f"DataLoader.__init__ called with metadata={metadata is not None}")
        if metadata:
            logger.info(f"Metadata keys: {metadata.keys()}")

        self.metadata_traces = None
        if metadata and 'traces' in metadata:
            self.metadata_traces = metadata['traces']
            logger.info(f"Set metadata_traces with {len(self.metadata_traces)} traces")
        else:
            logger.info("metadata_traces NOT set - metadata is None or missing 'traces' key")

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
                return self._convert_to_signal_dict(data, is_pickle=False)
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
                # Pass is_pickle=True to skip length filtering
                return self._convert_to_signal_dict(dataframe, is_pickle=True)

            return dataframe
        except Exception as e:
            logger.error(f"Error loading pickle file: {e}")
            return None

    def _convert_to_signal_dict(
            self,
            df: pd.DataFrame,
            is_pickle: bool = False
    ) -> List[Dict[str, Any]]:
        """Convert dataframe to signal dictionary format."""
        logger.info(f"_convert_to_signal_dict called with {len(df)} rows")
        logger.info(f"self.metadata_traces is: {self.metadata_traces is not None}")

        aa_field = self._find_aa_field(df)
        segment_field = self._find_segment_field(df)

        if self.metadata_traces:
            logger.info(f"Applying metadata filtering for {len(self.metadata_traces)} traces")
            df = self._filter_by_metadata(df)
            logger.info(f"After filtering: {len(df)} rows remain")
        else:
            logger.info("No metadata filtering - metadata_traces is None or empty")

        # Skip length filtering for pickle files (they contain pre-segmented data)
        if not is_pickle and (self.min_signal_length or self.max_signal_length):
            df = self._filter_by_length(df, segment_field)
            logger.debug(f"After length filter: {len(df)} records")
        elif is_pickle and (self.min_signal_length or self.max_signal_length):
            logger.debug("Skipping length filter for pre-segmented pickle data")

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
        """Filter dataframe by metadata traces using metadata (peptide), channel, and variable_region (AA)."""
        logger.info(f"DataFrame shape before filtering: {df.shape}")

        filtered_dfs = []
        matched_count = 0

        for trace in self.metadata_traces:
            channel = float(trace['Channel'])  # Convert to float to match df
            peptide_group = trace['Run']  # e.g., "HDKER", "AVLIM"
            amino_acid = trace['AA']  # e.g., "D", "A", "W"

            # Match on: metadata (peptide) AND channel AND variable_region (AA)
            trace_data = df[
                (df['metadata'].str.startswith(peptide_group)) &
                (df['channel'] == channel) &
                (df['variable_region'] == amino_acid)
                ]

            if not trace_data.empty:
                matched_count += len(trace_data)
                logger.info(f"  ✓ Found {len(trace_data)} match(es) for {peptide_group} Ch{channel} AA={amino_acid}")
                filtered_dfs.append(trace_data)
            else:
                logger.warning(f"  ✗ No match for {peptide_group} Ch{channel} AA={amino_acid}")

        if filtered_dfs:
            result = pd.concat(filtered_dfs, ignore_index=True)
            logger.info(
                f"✓ Metadata filtering: {matched_count} traces matched from {len(self.metadata_traces)} specified")
            return result

        logger.warning("✗ No traces matched metadata filters!")
        return pd.DataFrame()

    def _filter_by_length(self, df: pd.DataFrame, segment_field: str) -> pd.DataFrame:
        """Filter dataframe by signal length."""
        if segment_field:
            df['signal_length'] = df[segment_field].apply(self._get_signal_length)

            if self.min_signal_length:
                df = df[df['signal_length'] >= self.min_signal_length]
            if self.max_signal_length:
                df = df[df['signal_length'] <= self.max_signal_length]

            df = df.drop('signal_length', axis=1)

        return df

    @staticmethod
    def _get_signal_length(signal_value: Any) -> int:
        
        try:
            if isinstance(signal_value, str):
                try:
                    parsed = ast.literal_eval(signal_value)
                except:
                    try:
                        parsed = json.loads(signal_value)
                    except:
                        parsed = signal_value.split(',')
                return len(parsed)
            elif isinstance(signal_value, (list, np.ndarray)):
                return len(signal_value)
            else:
                return 0
        except:
            return 0

def parse_signal_data(signal_value: Any) -> np.ndarray:
    """Parse signal data from various formats to numpy array."""
    if isinstance(signal_value, str):
        try:
            parsed = ast.literal_eval(signal_value)
            return np.array(parsed, dtype=np.float64)
        except:
            try:
                parsed = json.loads(signal_value)
                return np.array(parsed, dtype=np.float64)
            except:
                parsed = [float(x.strip()) for x in signal_value.split(',')]
                return np.array(parsed, dtype=np.float64)
    elif isinstance(signal_value, list):
        return np.array(signal_value, dtype=np.float64)
    elif isinstance(signal_value, np.ndarray):
        return signal_value.astype(np.float64)
    else:
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