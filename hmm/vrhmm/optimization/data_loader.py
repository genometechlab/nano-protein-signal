"""
Data loading utilities for optimization.

Loads profiles and test traces directly without CLI overhead.
"""

import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')


class OptimizationDataLoader:
    """
    Loads data for optimization without subprocess overhead.
    
    Handles:
    - Profile CSV/JSON loading
    - Signal pickle/CSV loading
    - Metadata filtering
    - Z-normalization of traces
    """
    
    def __init__(
        self,
        profile_file: str,
        signal_file: str,
        metadata_file: Optional[str] = None,
        min_signal_length: Optional[int] = None,
        max_signal_length: Optional[int] = None
    ):
        self.profile_file = Path(profile_file)
        self.signal_file = Path(signal_file)
        self.metadata_file = Path(metadata_file) if metadata_file else None
        self.min_signal_length = min_signal_length
        self.max_signal_length = max_signal_length
        
        self._profiles: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None
        self._traces: Optional[Dict[str, List[np.ndarray]]] = None
    
    def load_profiles(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """
        Load profile statistics from file.
        
        Returns:
            {aa: {state_idx: (mean, std)}} dictionary
        """
        if self._profiles is not None:
            return self._profiles
        
        logger.info(f"Loading profiles from {self.profile_file}")
        
        if self.profile_file.suffix == '.json':
            self._profiles = self._load_profiles_json()
        else:
            self._profiles = self._load_profiles_csv()
        
        logger.info(f"Loaded profiles for {len(self._profiles)} amino acids")
        
        return self._profiles
    
    def _load_profiles_json(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Load profiles from JSON file."""
        with open(self.profile_file) as f:
            data = json.load(f)
        
        profiles = {}
        
        # Handle different JSON formats
        if 'profiles' in data:
            # Format: {profiles: {aa: {states: {idx: {mean, std}}}}}
            for aa, aa_data in data['profiles'].items():
                if aa not in AMINO_ACIDS:
                    continue
                profiles[aa] = {}
                states = aa_data.get('states', aa_data)
                for idx, stats in states.items():
                    if isinstance(stats, dict):
                        profiles[aa][str(idx)] = (float(stats['mean']), float(stats['std']))
                    else:
                        profiles[aa][str(idx)] = (float(stats[0]), float(stats[1]))
        else:
            # Format: {aa: {idx: (mean, std)}} or {aa: {idx: [mean, std]}}
            for aa, aa_data in data.items():
                if aa not in AMINO_ACIDS:
                    continue
                profiles[aa] = {}
                for idx, stats in aa_data.items():
                    if isinstance(stats, dict):
                        profiles[aa][str(idx)] = (float(stats['mean']), float(stats['std']))
                    else:
                        profiles[aa][str(idx)] = (float(stats[0]), float(stats[1]))
        
        return profiles
    
    def _load_profiles_csv(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Load profiles from CSV file."""
        df = pd.read_csv(self.profile_file)
        
        required_cols = {'amino_acid', 'state', 'mean', 'std'}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Profile CSV missing columns: {missing}")
        
        profiles = {}
        for aa in df['amino_acid'].unique():
            if aa not in AMINO_ACIDS:
                continue
            
            aa_df = df[df['amino_acid'] == aa].sort_values('state')
            profiles[aa] = {
                str(int(row['state'])): (float(row['mean']), float(row['std']))
                for _, row in aa_df.iterrows()
            }
        
        return profiles
    
    def load_traces(self) -> Dict[str, List[np.ndarray]]:
        """
        Load and process test traces.
        
        Returns:
            {aa: [trace1, trace2, ...]} where each trace is z-normalized segment means
        """
        if self._traces is not None:
            return self._traces
        
        logger.info(f"Loading traces from {self.signal_file}")
        
        # Load raw data
        if self.signal_file.suffix in ['.pkl', '.pickle']:
            raw_data = self._load_pickle()
        else:
            raw_data = self._load_csv()
        
        logger.info(f"Loaded {len(raw_data)} raw records")
        
        # Apply metadata filter if provided
        if self.metadata_file and self.metadata_file.exists():
            raw_data = self._apply_metadata_filter(raw_data)
            logger.info(f"After metadata filtering: {len(raw_data)} records")
        
        # Group by amino acid and z-normalize
        self._traces = self._process_traces(raw_data)
        
        total_traces = sum(len(t) for t in self._traces.values())
        logger.info(f"Loaded {total_traces} traces for {len(self._traces)} amino acids")
        
        return self._traces
    
    def _load_pickle(self) -> List[Dict[str, Any]]:
        """Load data from pickle file."""
        with open(self.signal_file, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, pd.DataFrame):
            return data.to_dict('records')
        elif isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'data' in data:
            df = data['data']
            return df.to_dict('records') if isinstance(df, pd.DataFrame) else df
        else:
            raise ValueError(f"Unexpected pickle format: {type(data)}")
    
    def _load_csv(self) -> List[Dict[str, Any]]:
        """Load data from CSV file."""
        df = pd.read_csv(self.signal_file)
        return df.to_dict('records')
    
    def _apply_metadata_filter(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter data based on metadata file."""
        logger.info(f"Applying metadata filter from {self.metadata_file}")
        
        with open(self.metadata_file) as f:
            metadata = json.load(f)
        
        if 'traces' not in metadata:
            logger.warning("Metadata file has no 'traces' key, skipping filter")
            return data
        
        traces_to_keep = metadata['traces']
        logger.info(f"Metadata specifies {len(traces_to_keep)} traces to keep")
        
        # Debug: show first trace spec
        if traces_to_keep:
            logger.info(f"  Example trace spec: {traces_to_keep[0]}")
        
        # Debug: show first data record keys
        if data:
            logger.info(f"  Data record keys: {list(data[0].keys())[:10]}")
        
        filtered = []
        for record in data:
            for trace_spec in traces_to_keep:
                # Match by channel and AA
                record_channel = record.get('channel')
                spec_channel = trace_spec.get('Channel')
                
                # Handle potential type mismatches
                try:
                    channel_match = float(record_channel) == float(spec_channel)
                except (TypeError, ValueError):
                    channel_match = str(record_channel) == str(spec_channel)
                
                aa_field = record.get('variable_region') or record.get('aa') or record.get('label')
                aa_match = aa_field == trace_spec.get('AA')
                
                # Optional: match by full metadata or run
                if 'full_metadata' in trace_spec and trace_spec['full_metadata']:
                    meta_match = record.get('metadata') == trace_spec['full_metadata']
                else:
                    run = trace_spec.get('Run', '')
                    record_meta = str(record.get('metadata', ''))
                    meta_match = record_meta.startswith(run) if run else True
                
                # Optional: match by df_index
                if 'df_index' in trace_spec and trace_spec['df_index'] is not None:
                    idx_match = str(record.get('df_index', '')) == str(trace_spec['df_index'])
                else:
                    idx_match = True
                
                if channel_match and aa_match and meta_match and idx_match:
                    filtered.append(record)
                    break
        
        logger.info(f"Metadata filtering: matched {len(filtered)} of {len(traces_to_keep)} specified traces")
        
        if len(filtered) == 0:
            logger.warning("No traces matched! Check metadata format.")
            # Debug: try to find why
            if data and traces_to_keep:
                sample_record = data[0]
                sample_spec = traces_to_keep[0]
                logger.warning(f"  Sample record: channel={sample_record.get('channel')}, "
                              f"aa={sample_record.get('variable_region')}, "
                              f"metadata={str(sample_record.get('metadata', ''))[:50]}")
                logger.warning(f"  Sample spec: Channel={sample_spec.get('Channel')}, "
                              f"AA={sample_spec.get('AA')}, "
                              f"Run={sample_spec.get('Run')}")
        
        return filtered
    
    def _process_traces(self, data: List[Dict[str, Any]]) -> Dict[str, List[np.ndarray]]:
        """Process raw data into z-normalized traces grouped by AA."""
        traces = {aa: [] for aa in AMINO_ACIDS}
        
        # Find AA field
        aa_field = None
        for field in ['variable_region', 'aa', 'label']:
            if data and field in data[0]:
                aa_field = field
                break
        
        if aa_field is None:
            logger.warning("Could not find amino acid field in data")
            return traces
        
        # Find segment field
        segment_field = None
        for field in ['cleaned_segments', 'cleaned_segment', 'raw_segments', 'segments']:
            if data and field in data[0]:
                segment_field = field
                break
        
        if segment_field is None:
            logger.warning("Could not find segment field in data")
            return traces
        
        for record in data:
            aa = record.get(aa_field, '')
            if aa not in AMINO_ACIDS:
                continue
            
            raw_segments = record.get(segment_field)
            if raw_segments is None:
                continue
            
            try:
                # Parse segments
                means = self._extract_segment_means(raw_segments)
                
                if means is None or len(means) == 0:
                    continue
                
                # Apply length filter
                if self.min_signal_length and len(means) < self.min_signal_length:
                    continue
                if self.max_signal_length and len(means) > self.max_signal_length:
                    continue
                
                # Z-normalize
                z_means = self._z_normalize(means)
                
                traces[aa].append(z_means)
                
            except Exception as e:
                logger.debug(f"Error processing record: {e}")
                continue
        
        # Remove empty AAs
        traces = {aa: t for aa, t in traces.items() if t}
        
        return traces
    
    def _extract_segment_means(self, raw_segments: Any) -> Optional[np.ndarray]:
        """Extract segment means from various formats."""
        import ast
        
        # Handle string representation
        if isinstance(raw_segments, str):
            try:
                raw_segments = ast.literal_eval(raw_segments)
            except:
                try:
                    raw_segments = json.loads(raw_segments)
                except:
                    return None
        
        if not isinstance(raw_segments, (list, np.ndarray)):
            return None
        
        if len(raw_segments) == 0:
            return None
        
        # Check if pre-segmented (list of lists/arrays)
        if isinstance(raw_segments[0], (list, np.ndarray)):
            means = []
            for seg in raw_segments:
                if seg is not None:
                    seg_array = np.array(seg).flatten()
                    if len(seg_array) > 0:
                        means.append(float(np.mean(seg_array)))
            return np.array(means) if means else None
        else:
            # Already flat array of values
            return np.array(raw_segments, dtype=np.float64)
    
    def _z_normalize(self, values: np.ndarray) -> np.ndarray:
        """Z-normalize an array."""
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        
        if std < 1e-10:
            return values - mean
        
        return (values - mean) / std
    
    def get_aa_trace_counts(self) -> Dict[str, int]:
        """Get count of traces per amino acid."""
        traces = self.load_traces()
        return {aa: len(t) for aa, t in traces.items()}
    
    def get_profile_lengths(self) -> Dict[str, int]:
        """Get number of states per amino acid profile."""
        profiles = self.load_profiles()
        return {aa: len(p) for aa, p in profiles.items()}