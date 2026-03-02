from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy import stats
from collections import Counter
import numpy as np
import polars as pl
import logging
from pathlib import Path
import pandas as pd

# Set PDF backend to make text selectable
mpl.rcParams['pdf.fonttype'] = 42  # TrueType fonts
mpl.rcParams['ps.fonttype'] = 42   # For EPS files too

logger = logging.getLogger(__name__)


class HMMVisualizationPlotter:
    """Handles all visualization for HMM classification analysis."""
    
    def __init__(self, analyzer, style='whitegrid', palette='muted', context='notebook'):
        """Initialize plotter with seaborn styling."""
        self.analyzer = analyzer
        
        sns.set_style(style)
        sns.set_palette(palette)
        sns.set_context(context)
        
        print('Initialized visualization plotter')
    
    def plot_profile_skip_analysis(self, segment_range: Tuple[int, int] = (0, 34),
                               figsize: Tuple[int, int] = (16, 8),
                               use_normalized: bool = True,
                               bar_width: float = 0.8):
        """Plot profile with correct segment indexing based on emitting states."""
        print(f'\nCreating profile skip analysis plot')
        
        if self.analyzer.profile_df is None:
            print('Warning: No profile data loaded')
            return None
        
        if use_normalized:
            if self.analyzer.normalized_segment_data_dict is None:
                print('Warning: Normalized segment data not available.')
                return None
            segment_data = self.analyzer.normalized_segment_data_dict
            ylabel = 'Z-Score Normalized Signal Value'
        else:
            if self.analyzer.segment_data_dict is None:
                print('Warning: No segment data loaded')
                return None
            segment_data = self.analyzer.segment_data_dict
            ylabel = 'Signal Value'
        
        if self.analyzer.matched_data is None:
            print('Warning: No matched data available')
            return None
        
        start, end = segment_range
        state_positions = list(range(start, end + 1))
        target_aa = self.analyzer.target_aas[0]
        
        profile_data = {}
        for state_pos in state_positions:
            profile_row = self.analyzer.profile_df.filter(
                (pl.col('amino_acid') == target_aa) & 
                (pl.col('state') == state_pos)
            )
            if len(profile_row) > 0:
                profile_data[state_pos] = {
                    'mean': profile_row['mean'][0],
                    'std': profile_row['std'][0],
                    'var': profile_row['var'][0]
                }
        
        skip_analysis = self.analyzer.analyze_skip_patterns(segment_range)
        
        state_segments = {}
        for state_pos in state_positions:
            state_segments[state_pos] = {
                'matched': [],
                'skipped': []
            }
        
        for trace_id, analysis in skip_analysis.items():
            if trace_id not in segment_data:
                continue
            
            cleaned_segments = segment_data[trace_id]['cleaned_segments']
            circle_skipped = set(analysis['circle_skipped'])
            circle_path = analysis['circle_path']
            
            for state_pos in state_positions:
                target_state = f'Match_{state_pos}'
                
                segment_idx = self.analyzer.get_segment_index_from_path(circle_path, state_pos)
                
                if segment_idx == -1:
                    if state_pos in circle_skipped:
                        if state_pos < len(cleaned_segments):
                            segment_mean = np.mean(cleaned_segments[state_pos])
                            state_segments[state_pos]['skipped'].append(segment_mean)
                    continue
                
                if segment_idx >= len(cleaned_segments):
                    continue
                
                segment_mean = np.mean(cleaned_segments[segment_idx])
                
                if target_state in circle_path:
                    state_segments[state_pos]['matched'].append(segment_mean)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = {
            'profile': sns.color_palette()[0],
            'matched': sns.color_palette()[1],
            'skipped': sns.color_palette()[2]
        }
        
        from matplotlib.patches import Rectangle
        
        for state_pos in state_positions:
            if state_pos not in profile_data:
                continue
            
            mean = profile_data[state_pos]['mean']
            std = profile_data[state_pos]['std']
            
            rect = Rectangle(
                (state_pos - bar_width/2, mean - std),
                bar_width,
                2 * std,
                facecolor=colors['profile'],
                edgecolor='white',
                alpha=0.3,
                linewidth=1,
                zorder=1
            )
            ax.add_patch(rect)
            
            ax.plot([state_pos - bar_width/2, state_pos + bar_width/2], 
                   [mean, mean], 
                   color=colors['profile'], 
                   linewidth=2, 
                   alpha=0.8,
                   zorder=2)
        
        ax.plot([], [], color=colors['profile'], linewidth=2, 
               label='Profile mean', alpha=0.8)
        ax.fill_between([], [], [], color=colors['profile'], alpha=0.3, 
                       label='Profile ± 1 std')
        
        for state_pos in state_positions:
            if not state_segments[state_pos]['skipped']:
                continue
            
            matched_means = state_segments[state_pos]['matched']
            skipped_means = state_segments[state_pos]['skipped']
            
            jitter = 0.15
            
            if matched_means:
                x_matched = [state_pos - jitter] * len(matched_means)
                ax.scatter(x_matched, matched_means, color=colors['matched'],
                          s=50, alpha=0.7, edgecolors='white', linewidth=0.5,
                          label='Matched segments' if state_pos == state_positions[0] else '',
                          zorder=3)
            
            if skipped_means:
                x_skipped = [state_pos + jitter] * len(skipped_means)
                ax.scatter(x_skipped, skipped_means, color=colors['skipped'],
                          s=50, alpha=0.9, edgecolors='white', linewidth=0.5,
                          label='Skipped segments' if state_pos == state_positions[0] else '',
                          zorder=3)
        
        ax.set_xlabel('State Position', fontsize=13, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
        ax.set_title(f'Profile vs Segment Analysis: {target_aa} (Circle Test)\nShowing segment data only for states that were skipped',
                    fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='best', fontsize=10, frameon=True, fancybox=True, shadow=True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        print('Profile skip analysis plot created')
        
        return fig
    
    def plot_individual_skip_details(self, within_profile_skips: List[Dict[str, Any]],
                                     max_plots: int = 10,
                                     save_dir: str = 'figures/skip_details',
                                     gap_width: int = 50):
        """Create detailed individual plots with artificial gaps for skipped state profiles."""
        from matplotlib.patches import Rectangle
        
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        print(f'\nCreating {min(len(within_profile_skips), max_plots)} detailed skip plots...')
        
        target_aa = self.analyzer.target_aas[0]
        
        for idx, skip_info in enumerate(within_profile_skips[:max_plots]):
            trace_id = skip_info['trace_id']
            focus_state_pos = skip_info['state_pos']
            
            if trace_id not in self.analyzer.segment_data_dict:
                continue
            
            norm_data = self.analyzer.normalized_segment_data_dict[trace_id]
            circle_path = self.analyzer.matched_data[trace_id]['circle_test']['full_path']
            
            fig, ax = plt.subplots(figsize=(18, 6))
            
            normalized_segments = norm_data['cleaned_segments']
            
            emission_to_state = []
            for state in circle_path:
                if state.startswith('Match_') or state.startswith('Insert_'):
                    emission_to_state.append(state)
            
            x_position = 0
            prev_match_num = -1
            prev_state = None
            
            # Track boundary positions for different types
            skip_boundaries = []
            slip_boundaries = []
            self_loop_boundaries = []
            
            for seg_idx in range(len(normalized_segments)):
                if seg_idx >= len(emission_to_state):
                    break
                
                state = emission_to_state[seg_idx]
                segment = normalized_segments[seg_idx]
                seg_len = len(segment)
                
                if state.startswith('Match_'):
                    match_num = int(state.split('_')[1])
                    
                    # Detect skip (forward jump)
                    if prev_match_num >= 0 and match_num > prev_match_num + 1:
                        skipped_states = list(range(prev_match_num + 1, match_num))
                        num_skipped = len(skipped_states)
                        total_skip_width = gap_width * num_skipped
                        section_width = total_skip_width / num_skipped
                        
                        skip_boundaries.append(x_position)
                        
                        for skip_idx, skipped_num in enumerate(skipped_states):
                            section_start = x_position + (skip_idx * section_width)
                            section_end = x_position + ((skip_idx + 1) * section_width)
                            
                            profile_row = self.analyzer.profile_df.filter(
                                (pl.col('amino_acid') == target_aa) & 
                                (pl.col('state') == skipped_num)
                            )
                            
                            if len(profile_row) > 0:
                                profile_mean = profile_row['mean'][0]
                                profile_std = profile_row['std'][0]
                                
                                rect = Rectangle(
                                    (section_start, profile_mean - profile_std),
                                    section_end - section_start,
                                    2 * profile_std,
                                    facecolor='gray',
                                    edgecolor='darkgray',
                                    alpha=0.4,
                                    linewidth=1.5,
                                    zorder=4
                                )
                                ax.add_patch(rect)
                                
                                ax.hlines(profile_mean, section_start, section_end,
                                        colors='darkgray', linewidth=3, alpha=0.9, 
                                        linestyles='-', zorder=5)
                                
                                mid_point = (section_start + section_end) / 2
                                y_min, y_max = ax.get_ylim()
                                label_y = y_min + 0.05 * (y_max - y_min)
                                
                                if skipped_num == focus_state_pos:
                                    bbox_props = dict(
                                        boxstyle='round,pad=0.3',
                                        facecolor='white',
                                        edgecolor='red',
                                        linewidth=3,
                                        alpha=0.95
                                    )
                                else:
                                    bbox_props = dict(
                                        boxstyle='round,pad=0.3',
                                        facecolor='lightgray',
                                        edgecolor='darkgray',
                                        linewidth=1,
                                        alpha=0.85
                                    )
                                
                                ax.text(mid_point, label_y, f'{skipped_num}',
                                       ha='center', va='bottom', fontsize=9,
                                       color='darkgray', weight='bold', alpha=0.9,
                                       bbox=bbox_props, zorder=8)
                        
                        x_position += total_skip_width
                    
                    # Detect backslip (backward jump)
                    if prev_match_num >= 0 and match_num < prev_match_num:
                        slip_boundaries.append(x_position)
                    
                    # Detect self-loop (same state twice)
                    if prev_state == state:
                        self_loop_boundaries.append(x_position)
                    
                    # Plot matched segment
                    color = '#1f77b4' if match_num % 2 == 0 else '#ff7f0e'
                    
                    if match_num == focus_state_pos:
                        linewidth = 3.0
                        alpha = 1.0
                    else:
                        linewidth = 1.0
                        alpha = 0.7
                    
                    profile_row = self.analyzer.profile_df.filter(
                        (pl.col('amino_acid') == target_aa) & 
                        (pl.col('state') == match_num)
                    )
                    
                    if len(profile_row) > 0:
                        profile_mean = profile_row['mean'][0]
                        profile_std = profile_row['std'][0]
                        
                        ax.fill_between(
                            x_position + np.arange(seg_len),
                            profile_mean - profile_std,
                            profile_mean + profile_std,
                            color=color,
                            alpha=0.2,
                            linewidth=0,
                            zorder=5
                        )
                        
                        ax.hlines(profile_mean, x_position, x_position + seg_len,
                                colors=color, linewidth=1.5, alpha=0.6, zorder=6)
                    
                    ax.plot(x_position + np.arange(seg_len), segment,
                           color=color, alpha=alpha, linewidth=linewidth, zorder=7)
                    
                    mid_point = x_position + seg_len / 2
                    y_min, y_max = ax.get_ylim()
                    label_y = y_min + 0.05 * (y_max - y_min)
                    
                    ax.text(mid_point, label_y, f'{match_num}',
                           ha='center', va='bottom', fontsize=10,
                           color=color, weight='bold', alpha=0.9, zorder=8)
                    
                    prev_match_num = match_num
                    prev_state = state
                    x_position += seg_len
                
                elif state.startswith('Insert_'):
                    ax.plot(x_position + np.arange(seg_len), segment,
                           color='green', alpha=0.7, linewidth=1.0, zorder=7)
                    
                    mid_point = x_position + seg_len / 2
                    y_min, y_max = ax.get_ylim()
                    label_y = y_min + 0.05 * (y_max - y_min)
                    
                    ax.text(mid_point, label_y, 'I',
                           ha='center', va='bottom', fontsize=8,
                           color='green', weight='bold', alpha=0.8, zorder=8)
                    
                    prev_state = state
                    x_position += seg_len
            
            # Draw boundary lines
            for skip_pos in skip_boundaries:
                ax.axvline(x=skip_pos, color='red', linestyle='--', 
                          linewidth=2, alpha=0.8, zorder=15)
            
            for slip_pos in slip_boundaries:
                ax.axvline(x=slip_pos, color='blue', linestyle='--', 
                          linewidth=2, alpha=0.8, zorder=15)
            
            for loop_pos in self_loop_boundaries:
                ax.axvline(x=loop_pos, color='magenta', linestyle=':', 
                          linewidth=2, alpha=0.8, zorder=15)
            
            profile_mean = skip_info['profile_mean']
            profile_std = skip_info['profile_std']
            segment_mean = skip_info['segment_mean']
            
            # Find which state this segment was actually aligned to
            aligned_to_state = None
            if focus_state_pos < len(emission_to_state):
                st = emission_to_state[focus_state_pos]
                if st.startswith('Match_'):
                    aligned_to_state = int(st.split('_')[1])
            
            ax.set_title(f'Skipped State Detail: {trace_id}\n'
                        f'Match_{focus_state_pos} skipped (pre-segment {focus_state_pos} aligned to Match_{aligned_to_state if aligned_to_state else "?"}) - within profile (deviation: {skip_info["deviation"]:.2f}σ)',
                        fontsize=13, fontweight='bold')
            ax.set_xlabel('Position (with artificial gaps for skipped states)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Z-Score Normalized Signal', fontsize=11, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            legend_text = f"Match_{focus_state_pos} Profile: μ={profile_mean:.2f}, σ={profile_std:.2f}\n"
            legend_text += f"Pre-segment {focus_state_pos} Mean: {segment_mean:.2f}\n"
            legend_text += f"Blue/Orange = Matched segments | Green = Insert states\n"
            legend_text += f"Gray bands = Skipped Match state profiles\n"
            legend_text += f"Red dashed = Skip | Blue dashed = Backslip | Magenta dotted = Self-loop"
            
            ax.text(0.02, 0.98, legend_text, transform=ax.transAxes,
                   fontsize=9, ha='left', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            plt.tight_layout()
            
            save_path = f'{save_dir}/skip_detail_{trace_id}_state{focus_state_pos}.pdf'
            plt.savefig(save_path, dpi=200, bbox_inches='tight', format='pdf')
            plt.close()
            
            print(f'  Saved: {save_path}')
        
        print(f'\nCompleted {min(len(within_profile_skips), max_plots)} detail plots')
    
    def plot_confusion_heatmap(self, figsize: Tuple[int, int] = (10, 8)):
        """Create a heatmap showing the confusion matrix of misclassifications."""
        print('\nCreating confusion heatmap')
        
        confusion_pairs = self.analyzer.get_confusion_pairs()
        
        all_aas = sorted(set([pair[0] for pair in confusion_pairs.keys()] + 
                            [pair[1] for pair in confusion_pairs.keys()]))
        
        matrix = np.zeros((len(all_aas), len(all_aas)))
        
        for (true_aa, pred_aa), traces in confusion_pairs.items():
            i = all_aas.index(true_aa)
            j = all_aas.index(pred_aa)
            matrix[i, j] = len(traces)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(matrix, annot=True, fmt='.0f', cmap='YlOrRd', 
                   xticklabels=all_aas, yticklabels=all_aas, 
                   cbar_kws={'label': 'Number of Traces'}, ax=ax,
                   linewidths=0.5, linecolor='gray')
        
        ax.set_xlabel('Predicted Amino Acid', fontsize=13, fontweight='bold')
        ax.set_ylabel('True Amino Acid', fontsize=13, fontweight='bold')
        ax.set_title('Misclassification Confusion Matrix', 
                    fontsize=14, fontweight='bold', pad=15)
        
        plt.tight_layout()
        print('Confusion heatmap created')
        
        return fig
    
    def plot_log_probability_comparison(self, figsize: Tuple[int, int] = (10, 6)):
        """Compare log probabilities between circle test and misclassified runs."""
        print('\nCreating log probability comparison plot')
        
        if self.analyzer.matched_data is None:
            print('Warning: No matched data available')
            return None
        
        circle_logprobs = []
        misclass_logprobs = []
        
        for trace_id, data in self.analyzer.matched_data.items():
            circle_logprobs.append(data['circle_test']['log_probability'])
            misclass_logprobs.append(data['misclassified']['log_probability'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        ax1.scatter(circle_logprobs, misclass_logprobs, alpha=0.6, 
                   s=50, edgecolors='white', linewidth=0.5)
        
        min_val = min(min(circle_logprobs), min(misclass_logprobs))
        max_val = max(max(circle_logprobs), max(misclass_logprobs))
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', 
                alpha=0.3, linewidth=2, label='y=x')
        
        ax1.set_xlabel('Circle Test Log Probability', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Misclassified Log Probability', fontsize=11, fontweight='bold')
        ax1.set_title('Log Probability Comparison', fontsize=12, fontweight='bold', pad=10)
        ax1.legend()
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        differences = [m - c for c, m in zip(circle_logprobs, misclass_logprobs)]
        ax2.hist(differences, bins=30, alpha=0.7, edgecolor='white', linewidth=1.5)
        ax2.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax2.set_xlabel('Log Probability Difference\n(Misclassified - Circle)', 
                      fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax2.set_title('Distribution of Differences', fontsize=12, fontweight='bold', pad=10)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        plt.tight_layout()
        print('Log probability comparison created')
        
        return fig

    def plot_kde_comparisons(self, kde_comparison_data: List[Dict[str, Any]],
                            max_plots: int = 10,
                            save_dir: str = 'figures/kde_comparisons'):
        """Create KDE comparison plots.
        
        Only for states that were:
        - Skipped in circle test (correct AA)
        - Matched in misclassified (wrong AA)
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        print(f'\nCreating {min(len(kde_comparison_data), max_plots)} KDE comparison plots...')
        
        for idx, data in enumerate(kde_comparison_data[:max_plots]):
            trace_id = data['trace_id']
            state_pos = data['state_pos']
            true_aa = data['true_aa']
            pred_aa = data['pred_aa']
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot 1: True AA profile (Gaussian)
            true_mean = data['true_profile_mean']
            true_std = data['true_profile_std']
            x_range_true = np.linspace(true_mean - 4*true_std, true_mean + 4*true_std, 200)
            true_gaussian = stats.norm.pdf(x_range_true, true_mean, true_std)
            
            ax.plot(x_range_true, true_gaussian, color='green', linewidth=2.5, 
                   label=f'True ({true_aa}) Profile: μ={true_mean:.2f}, σ={true_std:.2f}',
                   alpha=0.8, zorder=3)
            ax.fill_between(x_range_true, true_gaussian, alpha=0.2, color='green', zorder=2)
            
            # Plot 2: Predicted AA profile (Gaussian)
            pred_mean = data['pred_profile_mean']
            pred_std = data['pred_profile_std']
            x_range_pred = np.linspace(pred_mean - 4*pred_std, pred_mean + 4*pred_std, 200)
            pred_gaussian = stats.norm.pdf(x_range_pred, pred_mean, pred_std)
            
            ax.plot(x_range_pred, pred_gaussian, color='red', linewidth=2.5,
                   label=f'Predicted ({pred_aa}) Profile: μ={pred_mean:.2f}, σ={pred_std:.2f}',
                   alpha=0.8, zorder=3)
            ax.fill_between(x_range_pred, pred_gaussian, alpha=0.2, color='red', zorder=2)
            
            # Plot 3: Actual segment KDE
            segment_values = data['segment_values']
            if len(segment_values) > 1:
                segment_kde = stats.gaussian_kde(segment_values)
                segment_mean = np.mean(segment_values)
                segment_std = np.std(segment_values)
                x_range_seg = np.linspace(min(segment_values), max(segment_values), 200)
                
                ax.plot(x_range_seg, segment_kde(x_range_seg), color='blue', linewidth=2.5,
                       label=f'Actual Segment: μ={segment_mean:.2f}, σ={segment_std:.2f}',
                       alpha=0.8, zorder=4)
                ax.fill_between(x_range_seg, segment_kde(x_range_seg), alpha=0.2, color='blue', zorder=2)
            
            ax.set_xlabel('Z-Score Normalized Value', fontsize=12, fontweight='bold')
            ax.set_ylabel('Density', fontsize=12, fontweight='bold')
            ax.set_title(f'KDE Comparison: {trace_id} - State {state_pos}\n'
                        f'Skipped in {true_aa} circle test, Matched in {pred_aa} misclassification',
                        fontsize=13, fontweight='bold')
            ax.legend(loc='best', fontsize=10, frameon=True, fancybox=True, shadow=True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            
            save_path = f'{save_dir}/kde_comparison_{trace_id}_state{state_pos}.pdf'
            plt.savefig(save_path, dpi=200, bbox_inches='tight', format='pdf')
            plt.close()
            
            print(f'  Saved: {save_path}')
        
        print(f'\nCompleted {min(len(kde_comparison_data), max_plots)} KDE comparison plots')

    def plot_detailed_path_comparison(self, kde_comparison_data: List[Dict[str, Any]],
                                      max_plots: int = 10,
                                      save_dir: str = 'figures/path_comparisons',
                                      gap_width: int = 50):
        """Plot both circle test and misclassified paths for comparison."""
        from matplotlib.patches import Rectangle
        
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        print(f'\nCreating {min(len(kde_comparison_data), max_plots)} path comparison plots...')
        
        for idx, data in enumerate(kde_comparison_data[:max_plots]):
            trace_id = data['trace_id']
            focus_state_pos = data['state_pos']
            true_aa = data['true_aa']
            pred_aa = data['pred_aa']
            
            if trace_id not in self.analyzer.segment_data_dict:
                continue
            
            norm_data = self.analyzer.normalized_segment_data_dict[trace_id]
            circle_path = self.analyzer.matched_data[trace_id]['circle_test']['full_path']
            misclass_path = self.analyzer.matched_data[trace_id]['misclassified']['full_path']
            
            # Get log probabilities
            circle_logprob = self.analyzer.matched_data[trace_id]['circle_test']['log_probability']
            misclass_logprob = self.analyzer.matched_data[trace_id]['misclassified']['log_probability']
            
            # Create 2 subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=False)
            
            normalized_segments = norm_data['cleaned_segments']
            
            # Plot both paths
            self._plot_single_path(
                ax1, normalized_segments, circle_path, true_aa, 
                focus_state_pos, 
                f'Circle Test - {true_aa} Profile (Log Prob: {circle_logprob:.2f})',
                gap_width
            )
            
            self._plot_single_path(
                ax2, normalized_segments, misclass_path, pred_aa, 
                focus_state_pos, 
                f'Misclassified - {pred_aa} Profile (Log Prob: {misclass_logprob:.2f})',
                gap_width
            )
            
            fig.suptitle(f'Path Comparison: {trace_id}\n'
                        f'Match_{focus_state_pos}: Skipped in {true_aa} (LogP: {circle_logprob:.2f}), '
                        f'Matched in {pred_aa} (LogP: {misclass_logprob:.2f})',
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            save_path = f'{save_dir}/path_comparison_{trace_id}_state{focus_state_pos}.pdf'
            plt.savefig(save_path, dpi=200, bbox_inches='tight', format='pdf')
            plt.close()
            
            print(f'  Saved: {save_path}')
        
        print(f'\nCompleted {min(len(kde_comparison_data), max_plots)} path comparison plots')

    def _plot_single_path(self, ax, normalized_segments, path, amino_acid, 
                         focus_state_pos, title, gap_width):
        """Helper function to plot a single HMM path."""
        from matplotlib.patches import Rectangle
        
        emission_to_state = []
        for state in path:
            if state.startswith('Match_') or state.startswith('Insert_'):
                emission_to_state.append(state)
        
        x_position = 0
        prev_match_num = -1
        prev_state = None
        
        skip_boundaries = []
        slip_boundaries = []
        self_loop_boundaries = []
        
        for seg_idx in range(len(normalized_segments)):
            if seg_idx >= len(emission_to_state):
                break
            
            state = emission_to_state[seg_idx]
            segment = normalized_segments[seg_idx]
            seg_len = len(segment)
            
            if state.startswith('Match_'):
                match_num = int(state.split('_')[1])
                
                # Detect skip
                if prev_match_num >= 0 and match_num > prev_match_num + 1:
                    skipped_states = list(range(prev_match_num + 1, match_num))
                    num_skipped = len(skipped_states)
                    total_skip_width = gap_width * num_skipped
                    section_width = total_skip_width / num_skipped
                    
                    skip_boundaries.append(x_position)
                    
                    for skip_idx, skipped_num in enumerate(skipped_states):
                        section_start = x_position + (skip_idx * section_width)
                        section_end = x_position + ((skip_idx + 1) * section_width)
                        
                        profile_row = self.analyzer.profile_df.filter(
                            (pl.col('amino_acid') == amino_acid) & 
                            (pl.col('state') == skipped_num)
                        )
                        
                        if len(profile_row) > 0:
                            profile_mean = profile_row['mean'][0]
                            profile_std = profile_row['std'][0]
                            
                            rect = Rectangle(
                                (section_start, profile_mean - profile_std),
                                section_end - section_start,
                                2 * profile_std,
                                facecolor='gray',
                                edgecolor='darkgray',
                                alpha=0.4,
                                linewidth=1.5,
                                zorder=4
                            )
                            ax.add_patch(rect)
                            
                            ax.hlines(profile_mean, section_start, section_end,
                                    colors='darkgray', linewidth=3, alpha=0.9, 
                                    linestyles='-', zorder=5)
                            
                            mid_point = (section_start + section_end) / 2
                            y_min, y_max = ax.get_ylim()
                            label_y = y_min + 0.05 * (y_max - y_min)
                            
                            if skipped_num == focus_state_pos:
                                bbox_props = dict(
                                    boxstyle='round,pad=0.3',
                                    facecolor='white',
                                    edgecolor='red',
                                    linewidth=3,
                                    alpha=0.95
                                )
                            else:
                                bbox_props = dict(
                                    boxstyle='round,pad=0.3',
                                    facecolor='lightgray',
                                    edgecolor='darkgray',
                                    linewidth=1,
                                    alpha=0.85
                                )
                            
                            ax.text(mid_point, label_y, f'{skipped_num}',
                                   ha='center', va='bottom', fontsize=9,
                                   color='darkgray', weight='bold', alpha=0.9,
                                   bbox=bbox_props, zorder=8)
                    
                    x_position += total_skip_width
                
                # Detect backslip
                if prev_match_num >= 0 and match_num < prev_match_num:
                    slip_boundaries.append(x_position)
                
                # Detect self-loop
                if prev_state == state:
                    self_loop_boundaries.append(x_position)
                
                # Plot matched segment
                color = '#1f77b4' if match_num % 2 == 0 else '#ff7f0e'
                
                if match_num == focus_state_pos:
                    linewidth = 3.0
                    alpha = 1.0
                else:
                    linewidth = 1.0
                    alpha = 0.7
                
                profile_row = self.analyzer.profile_df.filter(
                    (pl.col('amino_acid') == amino_acid) & 
                    (pl.col('state') == match_num)
                )
                
                if len(profile_row) > 0:
                    profile_mean = profile_row['mean'][0]
                    profile_std = profile_row['std'][0]
                    
                    ax.fill_between(
                        x_position + np.arange(seg_len),
                        profile_mean - profile_std,
                        profile_mean + profile_std,
                        color=color,
                        alpha=0.2,
                        linewidth=0,
                        zorder=5
                    )
                    
                    ax.hlines(profile_mean, x_position, x_position + seg_len,
                            colors=color, linewidth=1.5, alpha=0.6, zorder=6)
                
                ax.plot(x_position + np.arange(seg_len), segment,
                       color=color, alpha=alpha, linewidth=linewidth, zorder=7)
                
                mid_point = x_position + seg_len / 2
                y_min, y_max = ax.get_ylim()
                label_y = y_min + 0.05 * (y_max - y_min)
                
                ax.text(mid_point, label_y, f'{match_num}',
                       ha='center', va='bottom', fontsize=10,
                       color=color, weight='bold', alpha=0.9, zorder=8)
                
                prev_match_num = match_num
                prev_state = state
                x_position += seg_len
            
            elif state.startswith('Insert_'):
                ax.plot(x_position + np.arange(seg_len), segment,
                       color='green', alpha=0.7, linewidth=1.0, zorder=7)
                
                mid_point = x_position + seg_len / 2
                y_min, y_max = ax.get_ylim()
                label_y = y_min + 0.05 * (y_max - y_min)
                
                ax.text(mid_point, label_y, 'I',
                       ha='center', va='bottom', fontsize=8,
                       color='green', weight='bold', alpha=0.8, zorder=8)
                
                prev_state = state
                x_position += seg_len
        
        # Draw boundaries
        for skip_pos in skip_boundaries:
            ax.axvline(x=skip_pos, color='red', linestyle='--', 
                      linewidth=2, alpha=0.8, zorder=15)
        
        for slip_pos in slip_boundaries:
            ax.axvline(x=slip_pos, color='blue', linestyle='--', 
                      linewidth=2, alpha=0.8, zorder=15)
        
        for loop_pos in self_loop_boundaries:
            ax.axvline(x=loop_pos, color='magenta', linestyle=':', 
                      linewidth=2, alpha=0.8, zorder=15)
        
        ax.set_title(title, fontsize=12, fontweight='bold', loc='left')
        ax.set_ylabel('Z-Score Normalized Signal', fontsize=11, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def plot_detailed_path_comparison_grouped(self, kde_comparison_data: List[Dict[str, Any]],
                                             save_dir: str = 'figures/path_comparisons',
                                             gap_width: int = 50):
        """Plot paths grouped by trace - one plot per trace showing all skipped states."""
        from matplotlib.patches import Rectangle
        from collections import defaultdict
        
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Group by trace_id
        traces_grouped = defaultdict(list)
        for data in kde_comparison_data:
            traces_grouped[data['trace_id']].append(data)
        
        print(f'\nCreating {len(traces_grouped)} grouped path comparison plots...')
        
        for trace_id, skip_data_list in traces_grouped.items():
            # Collect all focus states for this trace
            focus_states = [d['state_pos'] for d in skip_data_list]
            true_aa = skip_data_list[0]['true_aa']
            pred_aa = skip_data_list[0]['pred_aa']
            
            if trace_id not in self.analyzer.segment_data_dict:
                continue
            
            norm_data = self.analyzer.normalized_segment_data_dict[trace_id]
            circle_path = self.analyzer.matched_data[trace_id]['circle_test']['full_path']
            misclass_path = self.analyzer.matched_data[trace_id]['misclassified']['full_path']
            circle_logprob = self.analyzer.matched_data[trace_id]['circle_test']['log_probability']
            misclass_logprob = self.analyzer.matched_data[trace_id]['misclassified']['log_probability']
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=False)
            
            normalized_segments = norm_data['cleaned_segments']
            
            # Plot both paths, highlighting ALL focus states
            self._plot_single_path_multiple_focus(
                ax1, normalized_segments, circle_path, true_aa, 
                focus_states,
                f'Circle Test - {true_aa} Profile (Log Prob: {circle_logprob:.2f})',
                gap_width
            )
            
            self._plot_single_path_multiple_focus(
                ax2, normalized_segments, misclass_path, pred_aa, 
                focus_states,
                f'Misclassified - {pred_aa} Profile (Log Prob: {misclass_logprob:.2f})',
                gap_width
            )
            
            # Create title with all focus states
            focus_states_str = ', '.join([str(s) for s in sorted(focus_states)])
            fig.suptitle(f'Path Comparison: {trace_id}\n'
                        f'States {focus_states_str}: Skipped in {true_aa} (LogP: {circle_logprob:.2f}), '
                        f'Matched in {pred_aa} (LogP: {misclass_logprob:.2f})',
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # Use trace_id for filename
            save_path = f'{save_dir}/path_comparison_{trace_id}_states_{"_".join([str(s) for s in sorted(focus_states)])}.pdf'
            plt.savefig(save_path, dpi=200, bbox_inches='tight', format='pdf')
            plt.close()
            
            print(f'  Saved: {save_path} (highlighting states: {focus_states_str})')
        
        print(f'\nCompleted {len(traces_grouped)} grouped path comparison plots')

    def _plot_single_path_multiple_focus(self, ax, normalized_segments, path, amino_acid, 
                                         focus_states_list, title, gap_width):
        """Helper function to plot a single HMM path with MULTIPLE focus states highlighted."""
        from matplotlib.patches import Rectangle
        
        focus_states_set = set(focus_states_list)
        
        emission_to_state = []
        for state in path:
            if state.startswith('Match_') or state.startswith('Insert_'):
                emission_to_state.append(state)
        
        x_position = 0
        prev_match_num = -1
        prev_state = None
        
        skip_boundaries = []
        slip_boundaries = []
        self_loop_boundaries = []
        
        for seg_idx in range(len(normalized_segments)):
            if seg_idx >= len(emission_to_state):
                break
            
            state = emission_to_state[seg_idx]
            segment = normalized_segments[seg_idx]
            seg_len = len(segment)
            
            if state.startswith('Match_'):
                match_num = int(state.split('_')[1])
                
                # Detect skip
                if prev_match_num >= 0 and match_num > prev_match_num + 1:
                    skipped_states = list(range(prev_match_num + 1, match_num))
                    num_skipped = len(skipped_states)
                    total_skip_width = gap_width * num_skipped
                    section_width = total_skip_width / num_skipped
                    
                    skip_boundaries.append(x_position)
                    
                    for skip_idx, skipped_num in enumerate(skipped_states):
                        section_start = x_position + (skip_idx * section_width)
                        section_end = x_position + ((skip_idx + 1) * section_width)
                        
                        profile_row = self.analyzer.profile_df.filter(
                            (pl.col('amino_acid') == amino_acid) & 
                            (pl.col('state') == skipped_num)
                        )
                        
                        if len(profile_row) > 0:
                            profile_mean = profile_row['mean'][0]
                            profile_std = profile_row['std'][0]
                            
                            rect = Rectangle(
                                (section_start, profile_mean - profile_std),
                                section_end - section_start,
                                2 * profile_std,
                                facecolor='gray',
                                edgecolor='darkgray',
                                alpha=0.4,
                                linewidth=1.5,
                                zorder=4
                            )
                            ax.add_patch(rect)
                            
                            ax.hlines(profile_mean, section_start, section_end,
                                    colors='darkgray', linewidth=3, alpha=0.9, 
                                    linestyles='-', zorder=5)
                            
                            mid_point = (section_start + section_end) / 2
                            y_min, y_max = ax.get_ylim()
                            label_y = y_min + 0.05 * (y_max - y_min)
                            
                            # Highlight ALL focus states with red border
                            if skipped_num in focus_states_set:
                                bbox_props = dict(
                                    boxstyle='round,pad=0.3',
                                    facecolor='white',
                                    edgecolor='red',
                                    linewidth=3,
                                    alpha=0.95
                                )
                            else:
                                bbox_props = dict(
                                    boxstyle='round,pad=0.3',
                                    facecolor='lightgray',
                                    edgecolor='darkgray',
                                    linewidth=1,
                                    alpha=0.85
                                )
                            
                            ax.text(mid_point, label_y, f'{skipped_num}',
                                   ha='center', va='bottom', fontsize=9,
                                   color='darkgray', weight='bold', alpha=0.9,
                                   bbox=bbox_props, zorder=8)
                    
                    x_position += total_skip_width
                
                # Detect backslip
                if prev_match_num >= 0 and match_num < prev_match_num:
                    slip_boundaries.append(x_position)
                
                # Detect self-loop
                if prev_state == state:
                    self_loop_boundaries.append(x_position)
                
                # Plot matched segment
                color = '#1f77b4' if match_num % 2 == 0 else '#ff7f0e'
                
                # Highlight ALL focus states with thicker lines
                if match_num in focus_states_set:
                    linewidth = 3.0
                    alpha = 1.0
                else:
                    linewidth = 1.0
                    alpha = 0.7
                
                profile_row = self.analyzer.profile_df.filter(
                    (pl.col('amino_acid') == amino_acid) & 
                    (pl.col('state') == match_num)
                )
                
                if len(profile_row) > 0:
                    profile_mean = profile_row['mean'][0]
                    profile_std = profile_row['std'][0]
                    
                    ax.fill_between(
                        x_position + np.arange(seg_len),
                        profile_mean - profile_std,
                        profile_mean + profile_std,
                        color=color,
                        alpha=0.2,
                        linewidth=0,
                        zorder=5
                    )
                    
                    ax.hlines(profile_mean, x_position, x_position + seg_len,
                            colors=color, linewidth=1.5, alpha=0.6, zorder=6)
                
                ax.plot(x_position + np.arange(seg_len), segment,
                       color=color, alpha=alpha, linewidth=linewidth, zorder=7)
                
                mid_point = x_position + seg_len / 2
                y_min, y_max = ax.get_ylim()
                label_y = y_min + 0.05 * (y_max - y_min)
                
                ax.text(mid_point, label_y, f'{match_num}',
                       ha='center', va='bottom', fontsize=10,
                       color=color, weight='bold', alpha=0.9, zorder=8)
                
                prev_match_num = match_num
                prev_state = state
                x_position += seg_len
            
            elif state.startswith('Insert_'):
                ax.plot(x_position + np.arange(seg_len), segment,
                       color='green', alpha=0.7, linewidth=1.0, zorder=7)
                
                mid_point = x_position + seg_len / 2
                y_min, y_max = ax.get_ylim()
                label_y = y_min + 0.05 * (y_max - y_min)
                
                ax.text(mid_point, label_y, 'I',
                       ha='center', va='bottom', fontsize=8,
                       color='green', weight='bold', alpha=0.8, zorder=8)
                
                prev_state = state
                x_position += seg_len
        
        # Draw boundaries
        for skip_pos in skip_boundaries:
            ax.axvline(x=skip_pos, color='red', linestyle='--', 
                      linewidth=2, alpha=0.8, zorder=15)
        
        for slip_pos in slip_boundaries:
            ax.axvline(x=slip_pos, color='blue', linestyle='--', 
                      linewidth=2, alpha=0.8, zorder=15)
        
        for loop_pos in self_loop_boundaries:
            ax.axvline(x=loop_pos, color='magenta', linestyle=':', 
                      linewidth=2, alpha=0.8, zorder=15)
        
        ax.set_title(title, fontsize=12, fontweight='bold', loc='left')
        ax.set_ylabel('Z-Score Normalized Signal', fontsize=11, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)