from typing import Dict, List, Any, Tuple, Optional
import json
import polars as pl
import numpy as np
from collections import Counter
import logging
import pickle
import pandas as pd

def setup_logging(level=logging.INFO):
    """Call this at the top of your notebook with level=logging.DEBUG for detailed logs."""
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s',
        force=True
    )

logger = logging.getLogger(__name__)


class HMMClassificationAnalyzer:
    """Analyzes HMM classification results for amino acids."""
    
    def __init__(self, results_file: str, target_aas: List[str], 
                 save_keys: List[str] = None, verbose: bool = True):
        self.results_file = results_file
        self.target_aas = target_aas
        self.save_keys = save_keys or ['full_path', 'amino_acid', 
                                        'predicted_category', 'log_probability']
        self.results_dict = None
        self.filtered_results = None
        self.misclassified = None
        self.circle_test_results = None
        self.matched_data = None
        self.profile_df = None
        self.segment_data_dict = None
        self.normalized_segment_data_dict = None
        self.verbose = verbose
        
        if self.verbose:
            print(f'Initialized analyzer for amino acids: {target_aas}')

    def debug_emitting_state_counts(self) -> None:
        """Count Match and Insert states in all paths to verify they match segment count."""
        if self.matched_data is None:
            raise ValueError('Must have matched data')
        
        print('\nDebugging emitting state counts in paths...\n')
        
        results = []
        
        for trace_id, data in self.matched_data.items():
            circle_path = data['circle_test']['full_path']
            misclass_path = data['misclassified']['full_path']
            
            # Count emitting states in circle test
            circle_match_count = sum(1 for s in circle_path if s.startswith('Match_'))
            circle_insert_count = sum(1 for s in circle_path if s.startswith('Insert_'))
            circle_total = circle_match_count + circle_insert_count
            
            # Count emitting states in misclassified
            misclass_match_count = sum(1 for s in misclass_path if s.startswith('Match_'))
            misclass_insert_count = sum(1 for s in misclass_path if s.startswith('Insert_'))
            misclass_total = misclass_match_count + misclass_insert_count
            
            # Get actual segment count
            if trace_id in self.segment_data_dict:
                segment_count = len(self.segment_data_dict[trace_id]['cleaned_segments'])
            else:
                segment_count = None
            
            results.append({
                'trace_id': trace_id,
                'segments': segment_count,
                'circle_match': circle_match_count,
                'circle_insert': circle_insert_count,
                'circle_total': circle_total,
                'misclass_match': misclass_match_count,
                'misclass_insert': misclass_insert_count,
                'misclass_total': misclass_total
            })
        
        # Print summary
        print(f"{'Trace ID':<40} | Segs | Circle (M+I) | Misclass (M+I)")
        print("-" * 90)
        
        for r in results[:20]:  # Show first 20
            print(f"{r['trace_id']:<40} | {r['segments']:>4} | "
                  f"{r['circle_match']:>2}+{r['circle_insert']:>2}={r['circle_total']:>2} | "
                  f"{r['misclass_match']:>2}+{r['misclass_insert']:>2}={r['misclass_total']:>2}")
        
        # Statistical summary
        circle_totals = [r['circle_total'] for r in results]
        misclass_totals = [r['misclass_total'] for r in results]
        segment_counts = [r['segments'] for r in results if r['segments'] is not None]
        
        print("\nSummary Statistics:")
        print(f"Segment counts: min={min(segment_counts)}, max={max(segment_counts)}, "
              f"unique values={set(segment_counts)}")
        print(f"Circle test emitting states: min={min(circle_totals)}, max={max(circle_totals)}, "
              f"unique values={set(circle_totals)}")
        print(f"Misclass emitting states: min={min(misclass_totals)}, max={max(misclass_totals)}, "
              f"unique values={set(misclass_totals)}")
        
        # Check for discrepancies
        issues = []
        for r in results:
            if r['segments'] and r['circle_total'] != r['segments']:
                issues.append(f"{r['trace_id']}: {r['segments']} segments but {r['circle_total']} emitting states in circle")
        
        if issues:
            print(f"\nFound {len(issues)} traces with mismatches:")
            for issue in issues[:10]:
                print(f"  {issue}")
        else:
            print("\nAll traces have matching emitting state counts!")
        
    def load_and_clean(self) -> 'HMMClassificationAnalyzer':
        """Load results and filter for target amino acids."""
        if self.verbose:
            print(f'\nLoading results from {self.results_file}')
        
        with open(self.results_file, 'r') as f:
            results_dict = json.load(f)
        
        logger.debug(f'Total traces in file: {len(results_dict)}')
        
        self.results_dict = {
            key: value for key, value in results_dict.items()
            if key.split("_")[-1] in self.target_aas
        }
        
        if self.verbose:
            print(f'Filtered to {len(self.results_dict)} traces for target amino acids')
        
        sample_keys = list(self.results_dict.keys())[:3]
        logger.debug(f'Sample trace IDs: {sample_keys}')
        
        return self
    
    def filter_keys(self) -> 'HMMClassificationAnalyzer':
        """Keep only specified keys in each trace."""
        logger.info(f'Filtering to keep only keys: {self.save_keys}')
        
        self.filtered_results = {}
        for outer_key, inner_dict in self.results_dict.items():
            self.filtered_results[outer_key] = {
                k: inner_dict[k] for k in self.save_keys if k in inner_dict
            }
        
        if self.verbose:
            print(f'Filtered {len(self.filtered_results)} traces to keep keys: {self.save_keys}')
        
        if self.filtered_results:
            sample_trace = list(self.filtered_results.keys())[0]
            sample_keys = list(self.filtered_results[sample_trace].keys())
            logger.debug(f'Keys kept in sample trace {sample_trace}: {sample_keys}')
        
        return self
    
    def find_misclassified(self) -> 'HMMClassificationAnalyzer':
        """Find traces that were misclassified."""
        if self.verbose:
            print(f'\nFinding misclassified traces...')
        
        self.misclassified = {}
        for trace_id, trace_data in self.filtered_results.items():
            true_aa = trace_id.split('_')[-1]
            predicted_aa = trace_data['predicted_category']
            
            if predicted_aa != true_aa:
                self.misclassified[trace_id] = trace_data
                logger.debug(f'Misclassified: {trace_id} - True: {true_aa}, Predicted: {predicted_aa}')
        
        if self.verbose:
            print(f'Found {len(self.misclassified)} misclassified traces')
        
        confusion_counts = Counter()
        for trace_id, data in self.misclassified.items():
            true_aa = trace_id.split('_')[-1]
            pred_aa = data['predicted_category']
            confusion_counts[(true_aa, pred_aa)] += 1
        
        if self.verbose and confusion_counts:
            print('\nMisclassification distribution:')
            for (true_aa, pred_aa), count in sorted(confusion_counts.items()):
                print(f'  {true_aa} -> {pred_aa}: {count} traces')
        
        logger.debug(f'Misclassification distribution: {dict(confusion_counts)}')
        
        return self
    
    def load_circle_test(self, circle_test_file: str) -> 'HMMClassificationAnalyzer':
        """Load circle test results (amino acid vs its own profile)."""
        if self.verbose:
            print(f'\nLoading circle test results from {circle_test_file}')
        
        with open(circle_test_file, 'r') as f:
            circle_dict = json.load(f)
        
        logger.debug(f'Total traces in circle test file: {len(circle_dict)}')
        
        self.circle_test_results = {}
        for key, value in circle_dict.items():
            aa = key.split("_")[-1]
            if aa in self.target_aas:
                self.circle_test_results[key] = {
                    k: value[k] for k in self.save_keys if k in value
                }
        
        if self.verbose:
            print(f'Filtered to {len(self.circle_test_results)} circle test traces')
        
        sample_keys = list(self.circle_test_results.keys())[:3]
        logger.debug(f'Sample circle test trace IDs: {sample_keys}')
        
        return self
    
    def match_circle_to_misclassified(self) -> 'HMMClassificationAnalyzer':
        """Match misclassified traces with their circle test results."""
        if self.circle_test_results is None:
            raise ValueError('Must load circle test results first')
        if self.misclassified is None:
            raise ValueError('Must find misclassified traces first')
        
        if self.verbose:
            print(f'\nMatching misclassified traces with circle test data...')
        
        self.matched_data = {}
        unmatched_traces = []
        
        for trace_id in self.misclassified:
            if trace_id in self.circle_test_results:
                self.matched_data[trace_id] = {
                    'misclassified': self.misclassified[trace_id],
                    'circle_test': self.circle_test_results[trace_id]
                }
                logger.debug(f'Matched trace: {trace_id}')
            else:
                unmatched_traces.append(trace_id)
                logger.warning(f'No circle test data found for misclassified trace: {trace_id}')
        
        if self.verbose:
            print(f'Successfully matched {len(self.matched_data)} traces')
            if unmatched_traces:
                print(f'Warning: Could not match {len(unmatched_traces)} misclassified traces')
                print(f'First few unmatched: {unmatched_traces[:3]}')
        
        if self.matched_data:
            sample_id = list(self.matched_data.keys())[0]
            sample = self.matched_data[sample_id]
            circle_logprob = sample['circle_test']['log_probability']
            misclass_logprob = sample['misclassified']['log_probability']
            logger.debug(f'Sample {sample_id} - Circle log_prob: {circle_logprob:.2f}, '
                        f'Misclass log_prob: {misclass_logprob:.2f}')
            if self.verbose:
                print(f'\nExample log probability comparison:')
                print(f'  Trace: {sample_id}')
                print(f'  Circle test: {circle_logprob:.2f}')
                print(f'  Misclassified: {misclass_logprob:.2f}')
        
        return self
    
    def load_segment_data(self, segment_pickle_file: str) -> 'HMMClassificationAnalyzer':
        """Load segment data from pickle file and create trace_id mapping."""
        if self.verbose:
            print(f'\nLoading segment data from {segment_pickle_file}')
        
        with open(segment_pickle_file, 'rb') as f:
            segment_list = pickle.load(f)
        
        if self.verbose:
            print(f'Loaded {len(segment_list)} segment records')
        
        self.segment_data_dict = {}
        
        for record in segment_list:
            run = record.get('run', '')
            channel = record.get('channel', '')
            df_index = record.get('df_index', '')
            variable_region = record.get('variable_region', '')
            
            trace_id = f'{run}_{channel}_{df_index}_{variable_region}'
            
            self.segment_data_dict[trace_id] = {
                'cleaned_segments': record.get('cleaned_segments', []),
                'features': record.get('features', None),
                'label': record.get('label', None),
                'metadata': record.get('metadata', '')
            }
            
            logger.debug(f'Mapped trace_id: {trace_id}')
        
        if self.verbose:
            print(f'Created mapping for {len(self.segment_data_dict)} traces')
            print(f'Sample trace IDs: {list(self.segment_data_dict.keys())[:3]}')
        
        if self.misclassified:
            matched_count = sum(1 for tid in self.misclassified if tid in self.segment_data_dict)
            if self.verbose:
                print(f'{matched_count}/{len(self.misclassified)} misclassified traces have segment data')
        
        return self
    
    def zscore_normalize_segments(self) -> 'HMMClassificationAnalyzer':
        """Z-score normalize all segment data per trace."""
        if self.segment_data_dict is None:
            raise ValueError('Must load segment data first with load_segment_data()')
        
        if self.verbose:
            print('\nZ-score normalizing segment data...')
        
        self.normalized_segment_data_dict = {}
        
        for trace_id, data in self.segment_data_dict.items():
            cleaned_segments = data['cleaned_segments']
            
            all_values = []
            for segment in cleaned_segments:
                all_values.extend(segment)
            
            trace_mean = np.mean(all_values)
            trace_std = np.std(all_values)
            
            normalized_segments = []
            for segment in cleaned_segments:
                normalized_segment = [(val - trace_mean) / trace_std for val in segment]
                normalized_segments.append(normalized_segment)
            
            self.normalized_segment_data_dict[trace_id] = {
                'cleaned_segments': normalized_segments,
                'features': data['features'],
                'label': data['label'],
                'metadata': data['metadata'],
                'normalization_stats': {
                    'mean': trace_mean,
                    'std': trace_std
                }
            }
            
            logger.debug(f'Normalized {trace_id}: mean={trace_mean:.2f}, std={trace_std:.2f}')
        
        if self.verbose:
            print(f'Z-score normalized {len(self.normalized_segment_data_dict)} traces')
        
        return self

    def load_profiles(self, profile_file: str, load_all_aas: bool = False) -> 'HMMClassificationAnalyzer':
        """Load amino acid profile data from CSV.
        
        Args:
            profile_file: Path to profile CSV
            load_all_aas: If True, load all amino acids. If False, only load target_aas
        """
        if self.verbose:
            print(f'\nLoading profiles from {profile_file}')
        
        lf = pl.scan_csv(profile_file)
        
        if load_all_aas:
            self.profile_df = lf.collect()
        else:
            lf_filtered = lf.filter(pl.col('amino_acid').is_in(self.target_aas))
            self.profile_df = lf_filtered.collect()
        
        if self.verbose:
            print(f'Loaded {len(self.profile_df)} profile rows')
            print(f'Columns: {self.profile_df.columns}')
            
            unique_aas = self.profile_df['amino_acid'].unique().to_list()
            print(f'Amino acids in profiles: {sorted(unique_aas)}')
        
        logger.debug(f'Profile columns: {self.profile_df.columns}')
        
        if len(self.profile_df) > 0:
            logger.debug(f'Sample profile data:\n{self.profile_df.head(3)}')
        
        return self
    
    def find_skipped_states(self, path: List[str], 
                           segment_range: Tuple[int, int] = (10, 25)) -> List[int]:
        """Find which states in the segment range were skipped in the path."""
        start, end = segment_range
        expected_states = set(range(start, end + 1))
        
        visited_states = set()
        for state in path:
            if state.startswith('Match_'):
                state_num = int(state.split('_')[1])
                visited_states.add(state_num)
        
        skipped = sorted(expected_states - visited_states)
        
        logger.debug(f'Expected states in range {segment_range}: {sorted(expected_states)}')
        logger.debug(f'Visited states in range: {sorted(visited_states & expected_states)}')
        logger.debug(f'Skipped states: {skipped}')
        
        return skipped
    
    def analyze_skip_patterns(self, segment_range: Tuple[int, int] = (10, 25)
                             ) -> Dict[str, Dict[str, Any]]:
        """Analyze state skip patterns in matched data."""
        if self.matched_data is None:
            raise ValueError('Must match data first')
        
        if self.verbose:
            print(f'\nAnalyzing skip patterns for segment range {segment_range}')
        
        skip_analysis = {}
        for trace_id, data in self.matched_data.items():
            logger.debug(f'\nAnalyzing trace: {trace_id}')
            
            circle_path = data['circle_test']['full_path']
            misclass_path = data['misclassified']['full_path']
            
            true_aa = trace_id.split('_')[-1]
            predicted_aa = data['misclassified']['predicted_category']
            
            logger.debug(f'True AA: {true_aa}, Predicted AA: {predicted_aa}')
            logger.debug(f'Circle test path length: {len(circle_path)}')
            logger.debug(f'Misclassified path length: {len(misclass_path)}')
            
            circle_skipped = self.find_skipped_states(circle_path, segment_range)
            logger.debug(f'Circle test skipped states: {circle_skipped}')
            
            misclass_skipped = self.find_skipped_states(misclass_path, segment_range)
            logger.debug(f'Misclassified skipped states: {misclass_skipped}')
            
            both_skipped = list(set(circle_skipped) & set(misclass_skipped))
            logger.debug(f'States skipped in both: {both_skipped}')
            
            skip_analysis[trace_id] = {
                'circle_skipped': circle_skipped,
                'misclass_skipped': misclass_skipped,
                'both_skipped': both_skipped,
                'true_aa': true_aa,
                'predicted_aa': predicted_aa,
                'circle_path': circle_path,
                'misclass_path': misclass_path
            }
        
        if self.verbose:
            print(f'Completed skip pattern analysis for {len(skip_analysis)} traces')
            
            total_circle_skips = sum(len(d['circle_skipped']) for d in skip_analysis.values())
            total_misclass_skips = sum(len(d['misclass_skipped']) for d in skip_analysis.values())
            print(f'Total states skipped in circle tests: {total_circle_skips}')
            print(f'Total states skipped in misclassified: {total_misclass_skips}')
        
        return skip_analysis
    
    def get_segment_index_from_path(self, path: List[str], target_state_position: int) -> int:
        """Get the segment index for a given Match state position by counting emitting states."""
        target_state = f'Match_{target_state_position}'
        segment_index = 0
        
        for state in path:
            if state == target_state:
                return segment_index
            
            if state.startswith('Match_') or state.startswith('Insert_'):
                segment_index += 1
        
        return -1
    
    def debug_skip_state_context(self, trace_id: str, state_position: int, context: int = 2) -> None:
        """Print full_path context for both circle test and misclassified paths."""
        if self.matched_data is None or trace_id not in self.matched_data:
            print(f"Trace {trace_id} not found in matched data")
            return
        
        circle_path = self.matched_data[trace_id]['circle_test']['full_path']
        misclass_path = self.matched_data[trace_id]['misclassified']['full_path']
        
        target_state = f'Match_{state_position}'
        
        print(f"Debug for {trace_id} at Match state {state_position}")
        print(f"Target state: {target_state}\n")
        
        print("CIRCLE TEST PATH (where state was potentially skipped):")
        print("-" * 60)
        if target_state in circle_path:
            idx = circle_path.index(target_state)
            start = max(0, idx - context)
            end = min(len(circle_path), idx + context + 1)
            
            for i in range(start, end):
                marker = " >>> " if i == idx else "     "
                print(f"{marker}Path index {i}: {circle_path[i]}")
        else:
            print(f"  {target_state} NOT IN PATH (was skipped)")
            
            match_indices = [(i, s) for i, s in enumerate(circle_path) if s.startswith('Match_')]
            before_states = [(i, s) for i, s in match_indices if int(s.split('_')[1]) < state_position]
            after_states = [(i, s) for i, s in match_indices if int(s.split('_')[1]) > state_position]
            
            if before_states:
                print(f"\n  Last match before skip:")
                idx, state = before_states[-1]
                for i in range(max(0, idx - context), min(len(circle_path), idx + context + 1)):
                    marker = " >>> " if i == idx else "     "
                    print(f"{marker}Path index {i}: {circle_path[i]}")
            
            if after_states:
                print(f"\n  First match after skip:")
                idx, state = after_states[0]
                for i in range(max(0, idx - context), min(len(circle_path), idx + context + 1)):
                    marker = " >>> " if i == idx else "     "
                    print(f"{marker}Path index {i}: {circle_path[i]}")
        
        print("\n" + "MISCLASSIFIED PATH (where state was potentially matched):")
        print("-" * 60)
        if target_state in misclass_path:
            idx = misclass_path.index(target_state)
            start = max(0, idx - context)
            end = min(len(misclass_path), idx + context + 1)
            
            for i in range(start, end):
                marker = " >>> " if i == idx else "     "
                print(f"{marker}Path index {i}: {misclass_path[i]}")
        else:
            print(f"  {target_state} NOT IN PATH")
        
        print("\n")
    
    def validate_segment_coverage(self, segment_range: Tuple[int, int] = (0, 34)) -> Dict[int, Dict[str, int]]:
        """Validate that all states have the expected number of segment data points."""
        if self.matched_data is None:
            raise ValueError('Must match data first')
        
        print(f'\nValidating segment coverage for range {segment_range}')
        
        start, end = segment_range
        state_positions = list(range(start, end + 1))
        
        state_counts = {}
        for state_pos in state_positions:
            state_counts[state_pos] = {
                'matched': 0,
                'skipped': 0,
                'total': 0
            }
        
        skip_analysis = self.analyze_skip_patterns(segment_range)
        expected_count = len(skip_analysis)
        
        for trace_id, analysis in skip_analysis.items():
            circle_path = analysis['circle_path']
            circle_skipped = set(analysis['circle_skipped'])
            
            for state_pos in state_positions:
                target_state = f'Match_{state_pos}'
                
                if target_state in circle_path:
                    state_counts[state_pos]['matched'] += 1
                    state_counts[state_pos]['total'] += 1
                elif state_pos in circle_skipped:
                    state_counts[state_pos]['skipped'] += 1
                    state_counts[state_pos]['total'] += 1
        
        print(f'\nExpected count per state: {expected_count} traces')
        print('\nState coverage:')
        
        issues_found = False
        for state_pos in state_positions:
            counts = state_counts[state_pos]
            status = "✓" if counts['total'] == expected_count else "✗"
            print(f"{status} State {state_pos}: {counts['matched']} matched + {counts['skipped']} skipped = {counts['total']} total")
            
            if counts['total'] != expected_count:
                issues_found = True
                print(f"    WARNING: Expected {expected_count}, got {counts['total']}")
        
        if not issues_found:
            print(f'\nAll states have correct coverage: {expected_count} traces each')
        else:
            print(f'\nWARNING: Some states have incorrect coverage')
        
        return state_counts
    
    def find_skips_within_profile(self, segment_range: Tuple[int, int] = (0, 34)) -> List[Dict[str, Any]]:
        """Find skipped states where the segment value falls within the profile's std."""
        print('\nFinding skipped states within profile standard deviation...')
        
        if self.profile_df is None or self.normalized_segment_data_dict is None:
            print('Warning: Need both profile and normalized segment data')
            return []
        
        target_aa = self.target_aas[0]
        start, end = segment_range
        state_positions = list(range(start, end + 1))
        
        profile_data = {}
        for state_pos in state_positions:
            profile_row = self.profile_df.filter(
                (pl.col('amino_acid') == target_aa) & 
                (pl.col('state') == state_pos)
            )
            if len(profile_row) > 0:
                profile_data[state_pos] = {
                    'mean': profile_row['mean'][0],
                    'std': profile_row['std'][0]
                }
        
        skip_analysis = self.analyze_skip_patterns(segment_range)
        
        within_profile_skips = []
        
        for trace_id, analysis in skip_analysis.items():
            if trace_id not in self.normalized_segment_data_dict:
                continue
            
            circle_skipped = analysis['circle_skipped']
            if not circle_skipped:
                continue
            
            cleaned_segments = self.normalized_segment_data_dict[trace_id]['cleaned_segments']
            
            for state_pos in circle_skipped:
                if state_pos not in profile_data:
                    continue
                
                if state_pos >= len(cleaned_segments):
                    continue
                
                segment_mean = np.mean(cleaned_segments[state_pos])
                profile_mean = profile_data[state_pos]['mean']
                profile_std = profile_data[state_pos]['std']
                
                if abs(segment_mean - profile_mean) <= profile_std:
                    within_profile_skips.append({
                        'trace_id': trace_id,
                        'state_pos': state_pos,
                        'segment_mean': segment_mean,
                        'profile_mean': profile_mean,
                        'profile_std': profile_std,
                        'deviation': abs(segment_mean - profile_mean) / profile_std
                    })
        
        if self.verbose:
            print(f'Found {len(within_profile_skips)} skipped states within profile std')
        
        return within_profile_skips
    
    def get_misclassified_by_predicted(self, predicted_aa: str) -> Dict[str, Dict[str, Any]]:
        """Get all traces misclassified as a specific amino acid."""
        results = {
            trace_id: data for trace_id, data in self.misclassified.items()
            if data['predicted_category'] == predicted_aa
        }
        if self.verbose:
            print(f'Found {len(results)} traces misclassified as {predicted_aa}')
        return results
    
    def get_confusion_pairs(self) -> Dict[tuple, List[str]]:
        """Group misclassifications by (true_aa, predicted_aa) pairs."""
        confusion_pairs = {}
        for trace_id, data in self.misclassified.items():
            true_aa = trace_id.split('_')[-1]
            predicted_aa = data['predicted_category']
            pair = (true_aa, predicted_aa)
            
            if pair not in confusion_pairs:
                confusion_pairs[pair] = []
            confusion_pairs[pair].append(trace_id)
        
        logger.info(f'Found {len(confusion_pairs)} unique confusion pairs')
        for pair, traces in confusion_pairs.items():
            logger.debug(f'  {pair[0]} -> {pair[1]}: {len(traces)} traces')
        
        return confusion_pairs

    def prepare_kde_comparison_data(self, within_profile_skips: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare data for KDE comparison plots."""
        print('\nPreparing KDE comparison data...')
        
        if self.normalized_segment_data_dict is None:
            raise ValueError('Need normalized segment data')
        if self.profile_df is None:
            raise ValueError('Need profile data')
        
        kde_comparison_data = []
        
        for skip_info in within_profile_skips:
            trace_id = skip_info['trace_id']
            state_pos = skip_info['state_pos']
            
            if trace_id not in self.normalized_segment_data_dict:
                continue
            
            # Get the true and predicted amino acids
            true_aa = trace_id.split('_')[-1]
            pred_aa = self.matched_data[trace_id]['misclassified']['predicted_category']
            
            # Get segment data at this position
            normalized_segments = self.normalized_segment_data_dict[trace_id]['cleaned_segments']
            if state_pos >= len(normalized_segments):
                continue
            
            segment_values = normalized_segments[state_pos]
            
            # Get true AA profile at this state
            true_profile = self.profile_df.filter(
                (pl.col('amino_acid') == true_aa) & 
                (pl.col('state') == state_pos)
            )
            
            # Get predicted AA profile at this state
            pred_profile = self.profile_df.filter(
                (pl.col('amino_acid') == pred_aa) & 
                (pl.col('state') == state_pos)
            )
            
            if len(true_profile) == 0 or len(pred_profile) == 0:
                continue
            
            kde_comparison_data.append({
                'trace_id': trace_id,
                'state_pos': state_pos,
                'true_aa': true_aa,
                'pred_aa': pred_aa,
                'segment_values': segment_values,
                'true_profile_mean': true_profile['mean'][0],
                'true_profile_std': true_profile['std'][0],
                'pred_profile_mean': pred_profile['mean'][0],
                'pred_profile_std': pred_profile['std'][0],
                'deviation': skip_info['deviation']
            })
        
        print(f'Prepared KDE data for {len(kde_comparison_data)} skipped states')
        
        return kde_comparison_data

    def prepare_kde_comparison_data(self, within_profile_skips: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare KDE data ONLY for states skipped in circle but matched in misclassified."""
        print('\nPreparing KDE comparison data...')
        
        if self.normalized_segment_data_dict is None:
            raise ValueError('Need normalized segment data')
        if self.profile_df is None:
            raise ValueError('Need profile data')
        
        kde_comparison_data = []
        
        for skip_info in within_profile_skips:
            trace_id = skip_info['trace_id']
            state_pos = skip_info['state_pos']
            
            if trace_id not in self.normalized_segment_data_dict:
                continue
            
            # Check if this state was matched in misclassified path
            misclass_path = self.matched_data[trace_id]['misclassified']['full_path']
            target_state = f'Match_{state_pos}'
            
            if target_state not in misclass_path:
                # State was also skipped in misclassified - skip this one
                logger.debug(f'Skipping {trace_id} state {state_pos} - also skipped in misclassified')
                continue
            
            # This state was skipped in circle but MATCHED in misclassified!
            true_aa = trace_id.split('_')[-1]
            pred_aa = self.matched_data[trace_id]['misclassified']['predicted_category']
            
            normalized_segments = self.normalized_segment_data_dict[trace_id]['cleaned_segments']
            if state_pos >= len(normalized_segments):
                continue
            
            segment_values = normalized_segments[state_pos]
            
            true_profile = self.profile_df.filter(
                (pl.col('amino_acid') == true_aa) & 
                (pl.col('state') == state_pos)
            )
            
            pred_profile = self.profile_df.filter(
                (pl.col('amino_acid') == pred_aa) & 
                (pl.col('state') == state_pos)
            )
            
            if len(true_profile) == 0 or len(pred_profile) == 0:
                continue
            
            kde_comparison_data.append({
                'trace_id': trace_id,
                'state_pos': state_pos,
                'true_aa': true_aa,
                'pred_aa': pred_aa,
                'segment_values': segment_values,
                'true_profile_mean': true_profile['mean'][0],
                'true_profile_std': true_profile['std'][0],
                'pred_profile_mean': pred_profile['mean'][0],
                'pred_profile_std': pred_profile['std'][0],
                'deviation': skip_info['deviation']
            })
        
        print(f'Prepared KDE data for {len(kde_comparison_data)} states (skipped in circle, matched in misclassified)')
        
        return kde_comparison_data

    def create_path_statistics_table(self, kde_comparison_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create a comparison table of path statistics for circle vs misclassified."""
        import pandas as pd
        
        print('\nCreating path statistics table...')
        
        # Group by trace first to get stats once per trace
        from collections import defaultdict
        traces_grouped = defaultdict(list)
        for data in kde_comparison_data:
            traces_grouped[data['trace_id']].append(data)
        
        table_data = []
        
        for trace_id, skip_data_list in traces_grouped.items():
            # Get path statistics once per trace
            circle_path = self.matched_data[trace_id]['circle_test']['full_path']
            misclass_path = self.matched_data[trace_id]['misclassified']['full_path']
            circle_logprob = self.matched_data[trace_id]['circle_test']['log_probability']
            misclass_logprob = self.matched_data[trace_id]['misclassified']['log_probability']
            
            circle_stats = self._count_path_events(circle_path)
            misclass_stats = self._count_path_events(misclass_path)
            
            true_aa = skip_data_list[0]['true_aa']
            pred_aa = skip_data_list[0]['pred_aa']
            
            # Collect all focus states for this trace
            focus_states = sorted([d['state_pos'] for d in skip_data_list])
            focus_states_str = ', '.join([str(s) for s in focus_states])
            
            # Create one row per trace
            table_data.append({
                'trace_id': trace_id,
                'num_focus_states': len(focus_states),
                'focus_states': focus_states_str,
                'true_aa': true_aa,
                'pred_aa': pred_aa,
                'circle_skips': circle_stats['skips'],
                'circle_slips': circle_stats['slips'],
                'circle_inserts': circle_stats['inserts'],
                'circle_self_loops': circle_stats['self_loops'],
                'circle_logprob': circle_logprob,
                'misclass_skips': misclass_stats['skips'],
                'misclass_slips': misclass_stats['slips'],
                'misclass_inserts': misclass_stats['inserts'],
                'misclass_self_loops': misclass_stats['self_loops'],
                'misclass_logprob': misclass_logprob,
                'logprob_diff': misclass_logprob - circle_logprob
            })
        
        df = pd.DataFrame(table_data)
        
        # Sort by number of focus states (descending) then by trace_id
        df = df.sort_values(['num_focus_states', 'trace_id'], ascending=[False, True])
        
        if self.verbose:
            print('\nPath Statistics Table (grouped by trace):')
            print(df.to_string(index=False))
            
            print(f'\nTraces with multiple skipped states:')
            multi_skip = df[df['num_focus_states'] > 1]
            if len(multi_skip) > 0:
                print(multi_skip[['trace_id', 'num_focus_states', 'focus_states', 'true_aa', 'pred_aa']].to_string(index=False))
            else:
                print('  None found')
        
        return df
    
    def _count_path_events(self, path: List[str]) -> Dict[str, int]:
        """Count skips, slips, inserts, and self-loops in a path."""
        counts = {
            'skips': 0,
            'slips': 0,
            'inserts': 0,
            'self_loops': 0
        }
        
        prev_match_num = -1
        prev_state = None
        
        for state in path:
            if state.startswith('Match_'):
                match_num = int(state.split('_')[1])
                
                if prev_match_num >= 0 and match_num > prev_match_num + 1:
                    counts['skips'] += 1
                
                if prev_match_num >= 0 and match_num < prev_match_num:
                    counts['slips'] += 1
                
                if state == prev_state:
                    counts['self_loops'] += 1
                
                prev_match_num = match_num
                prev_state = state
            
            elif state.startswith('Insert_'):
                counts['inserts'] += 1
                prev_state = state
        
        return counts