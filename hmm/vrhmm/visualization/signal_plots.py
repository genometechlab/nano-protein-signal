"""Signal visualization utilities for HMM analysis."""

"""Logging configuration to suppress noisy libraries."""

import logging
import warnings

def configure_logging():
    """Suppress verbose logging from matplotlib and related libraries."""
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    warnings.filterwarnings("ignore", category=FutureWarning, module="matplotlib")
    
    noisy_loggers = [
        'matplotlib',
        'matplotlib.font_manager',
        'matplotlib.backends',
        'fontTools',
        'fontTools.subset',
        'PIL',
        'PIL.PngImagePlugin'
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    # Even more aggressive for fontTools
    logging.getLogger('fontTools.subset').setLevel(logging.ERROR)

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Suppress matplotlib and fontTools logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('fontTools').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)
mpl.set_loglevel("ERROR")

logger = logging.getLogger(__name__)

def extract_channel_from_key(signal_key: str) -> str:
    """Extract channel information from signal key."""
    parts = signal_key.split('_')
    if len(parts) >= 2:
        return parts[1]
    return "unknown"

class HMMVisualizer:
    """Visualizer for HMM segmentation results with barycenter profile bands."""

    def __init__(self, custom_colors: Optional[List[str]] = None):
        
        # Load custom matplotlib style if available
        if 'genometechlab_main' in plt.style.available:
            plt.style.use('genometechlab_main')

        # Override problematic settings for compatibility
        mpl.rcParams['figure.autolayout'] = False
        mpl.rcParams['figure.constrained_layout.use'] = False

        # Set default colors for HMM visualization
        self.color_blue = '#1f77b4'
        self.color_orange = '#ff7f0e'

        # Keep custom_colors for pileup plots
        if custom_colors is None:
            self.custom_colors = [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
            ]
        else:
            self.custom_colors = custom_colors

        self.skip_color = '#ff0000'
        self.slip_line_color = '#0000ff'
        self.self_loop_color = '#ff00ff'

    def get_segment_color(self, segment_idx: int, total_segments: int = 35) -> Tuple[str, float]:
        
        color = self.color_blue if segment_idx % 2 == 0 else self.color_orange
        alpha_min = 0.3
        alpha_max = 1.0
        alpha = alpha_min + (alpha_max - alpha_min) * (segment_idx / max(1, total_segments - 1))
        return color, alpha

    def plot_hmm_segmentation(
            self,
            signal: np.ndarray,
            segment_results: Dict[str, Any],
            state_sequence: List[str],
            full_path: List[str],
            signal_key: str = "",
            save_path: Optional[str] = None,
            figsize: Tuple[int, int] = (8, 5.5)
    ) -> None:
        """Plot HMM segmentation with barycenter profile bands."""
        breakpoints = segment_results['breakpoints']
        means = segment_results['means']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # Top panel: NO HMM inference, just sequential segmentation
        self._plot_segmented_signal(ax1, signal, breakpoints, means, signal_key, segment_results)

        # Bottom panel: HMM-colored with match state information
        self._plot_hmm_colored_signal(ax2, signal, breakpoints, state_sequence,
                                      full_path, signal_key, segment_results)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    def _plot_segmented_signal(
            self,
            ax: plt.Axes,
            signal: np.ndarray,
            breakpoints: List[int],
            means: np.ndarray,
            title_suffix: str = "",
            segment_results: Optional[Dict[str, Any]] = None
    ) -> None:
        """Plot segmented signal WITHOUT HMM inference - just sequential segments."""
        channel = extract_channel_from_key(title_suffix)
        ax.set_title(f"Segmented Signal (No HMM) - Channel {channel}")
        ax.set_ylabel("Z-Normalized Current")

        num_segments = len(breakpoints) - 1

        for i in range(num_segments):
            start_idx = breakpoints[i]
            end_idx = breakpoints[i + 1]
            segment_range = np.arange(start_idx, end_idx)

            color, alpha = self.get_segment_color(i, num_segments)

            ax.plot(segment_range, signal[start_idx:end_idx],
                    color=color, alpha=alpha, linewidth=0.75, zorder=7)

            # Add sequential segment index label
            mid_point = (start_idx + end_idx) / 2
            y_limits = ax.get_ylim()
            if y_limits[0] != 0 or y_limits[1] != 1:
                label_y = y_limits[0] + 0.03 * (y_limits[1] - y_limits[0])
            else:
                label_y = -2.8

            ax.text(mid_point, label_y, str(i),
                    ha='center', va='bottom', fontsize=7,
                    color=color, weight='bold', alpha=min(alpha + 0.3, 1.0), zorder=8)

        ax.set_xlim(0, len(signal))

        # Force spines to be visible
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_linewidth(2.0)
        ax.spines['left'].set_linewidth(2.0)

    def _plot_hmm_colored_signal(
            self,
            ax: plt.Axes,
            signal: np.ndarray,
            breakpoints: List[int],
            state_sequence: List[str],
            full_path: List[str],
            title_suffix: str = "",
            segment_results: Optional[Dict[str, Any]] = None
    ) -> None:
        """Plot HMM state-colored signal with barycenter profile bands BY MATCH STATE."""
        channel = extract_channel_from_key(title_suffix)
        ax.set_title(f"HMM State-Colored Signal - Channel {channel}")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Z-Normalized Current")

        # Separate different state types from the path
        match_states_in_path = []
        insert_states_in_path = []
        segment_to_state = {}
        insert_boundaries = []

        obs_idx = 0
        for state in full_path:
            if 'Match' in state:
                if obs_idx < len(breakpoints) - 1:
                    match_states_in_path.append(state)
                    segment_to_state[obs_idx] = ('match', state)
                    obs_idx += 1
            elif 'Insert' in state:
                if obs_idx < len(breakpoints) - 1:
                    insert_states_in_path.append(state)
                    segment_to_state[obs_idx] = ('insert', state)
                    if obs_idx < len(breakpoints):
                        insert_boundaries.append(breakpoints[obs_idx])
                    obs_idx += 1

        # Find boundaries - RESTORED LOGIC
        skip_boundaries, slip_boundaries, self_loop_positions = [], [], []
        prev_state = None
        prev_match_idx = -1
        obs_count = 0

        for state in full_path:
            if 'Match' in state:
                curr_match_idx = int(state.split('_')[1])

                # Detect skip: jumped forward more than 1
                if prev_match_idx >= 0 and curr_match_idx > prev_match_idx + 1:
                    if obs_count < len(breakpoints):
                        skip_boundaries.append(breakpoints[obs_count])

                # Detect slip (backslip): jumped backward
                if prev_match_idx >= 0 and curr_match_idx < prev_match_idx:
                    if obs_count < len(breakpoints):
                        slip_boundaries.append(breakpoints[obs_count])

                # Detect self-loop: same state twice in a row
                if state == prev_state and 0 < obs_count < len(breakpoints):
                    self_loop_positions.append(breakpoints[obs_count])

                prev_match_idx = curr_match_idx
                obs_count += 1
                prev_state = state

        num_segments = len(breakpoints) - 1

        # Count unique match states
        unique_match_states = set()
        for idx, (state_type, state_name) in segment_to_state.items():
            if state_type == 'match':
                match_pos = int(state_name.split('_')[1])
                unique_match_states.add(match_pos)
        num_unique_matches = len(unique_match_states)

        # Plot segments
        for seg_idx in range(num_segments):
            start_idx = breakpoints[seg_idx]
            end_idx = breakpoints[seg_idx + 1]
            segment_range = np.arange(start_idx, end_idx)

            if seg_idx not in segment_to_state:
                # Skipped segment
                ax.plot(segment_range, signal[start_idx:end_idx],
                        color=self.skip_color, alpha=0.3, linewidth=0.5, zorder=7)
            else:
                state_type, state_name = segment_to_state[seg_idx]

                if state_type == 'insert':
                    # INSERT STATE SEGMENT - Use green color
                    insert_color = '#00ff00'

                    ax.plot(segment_range, signal[start_idx:end_idx],
                            color=insert_color, alpha=0.7, linewidth=0.75, zorder=8)

                    # Add insert state label
                    if '_' in state_name:
                        parts = state_name.split('_')
                        if len(parts) >= 3:
                            prev_idx = parts[1]
                            next_idx = parts[2]
                            label = f'i_{prev_idx}_{next_idx}'
                        else:
                            label = 'ins'
                    else:
                        label = 'ins'

                    mid_point = (start_idx + end_idx) / 2
                    y_limits = ax.get_ylim()
                    label_y = y_limits[0] + 0.03 * (y_limits[1] - y_limits[0])

                    ax.text(mid_point, label_y, label,
                            ha='center', va='bottom', fontsize=8,
                            color=insert_color, weight='bold',
                            style='italic', zorder=9)

                elif state_type == 'match':
                    # MATCH STATE SEGMENT
                    match_position = int(state_name.split('_')[1])
                    color, alpha_val = self.get_segment_color(match_position, num_segments)

                    # Profile bands if available
                    if segment_results and 'hmm_profile_stats' in segment_results and \
                            str(match_position) in segment_results['hmm_profile_stats']:
                        profile_mean = segment_results['hmm_profile_stats'][str(match_position)][0]
                        profile_std = segment_results['hmm_profile_stats'][str(match_position)][1]

                        ax.fill_between(
                            segment_range,
                            profile_mean - profile_std,
                            profile_mean + profile_std,
                            color=color,
                            alpha=alpha_val * 0.3,
                            linewidth=0,
                            zorder=5
                        )

                        ax.hlines(profile_mean, start_idx, end_idx,
                                  colors=color, linewidth=1.0, alpha=alpha_val * 0.8, zorder=6)

                    # Plot signal
                    ax.plot(segment_range, signal[start_idx:end_idx],
                            color=color, alpha=alpha_val, linewidth=0.50, zorder=7)

                    # Add match state label
                    mid_point = (start_idx + end_idx) / 2
                    y_limits = ax.get_ylim()
                    label_y = y_limits[0] + 0.03 * (y_limits[1] - y_limits[0])

                    ax.text(mid_point, label_y, f'{match_position}',
                            ha='center', va='bottom', fontsize=10,
                            color=color, weight='bold', alpha=min(alpha_val + 0.3, 1.0), zorder=8)

        # Add boundary lines
        for insert_pos in insert_boundaries:
            ax.axvline(x=insert_pos, color='green', linestyle=':', linewidth=1.5, alpha=0.8, zorder=15)

        for skip_pos in skip_boundaries:
            ax.axvline(x=skip_pos, color='red', linestyle='--', linewidth=0.75, alpha=0.9, zorder=15)

        for slip_pos in slip_boundaries:
            ax.axvline(x=slip_pos, color='blue', linestyle='--', linewidth=0.75, alpha=0.9, zorder=15)

        for self_loop_pos in self_loop_positions:
            ax.axvline(x=self_loop_pos, color='magenta', linestyle=':', linewidth=0.75, alpha=0.8, zorder=15)

        ax.set_xlim(0, len(signal))

        # Force spines to be visible
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_linewidth(2.0)
        ax.spines['left'].set_linewidth(2.0)

        # Add legend
        legend_elements = [
            mpatches.Patch(color=self.color_blue, alpha=0.6,
                           label=f'Match States: {num_unique_matches} unique (Blue/Orange)'),
        ]

        if insert_boundaries:
            legend_elements.append(mpatches.Patch(color='#00ff00', alpha=0.7,
                                                  label=f'Insert States ({len(insert_boundaries)})'))
            legend_elements.append(plt.Line2D([0], [0], color='green', linestyle=':', linewidth=1.5,
                                              label='Insert Boundaries'))

        if skip_boundaries:
            legend_elements.append(plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1.5,
                                              label=f'Skip Boundaries ({len(skip_boundaries)})'))
        if slip_boundaries:
            legend_elements.append(plt.Line2D([0], [0], color='blue', linestyle='--', linewidth=1.5,
                                              label=f'Slip Boundaries ({len(slip_boundaries)})'))
        if self_loop_positions:
            legend_elements.append(plt.Line2D([0], [0], color='magenta', linestyle=':', linewidth=2.0,
                                              label=f'Self-loops ({len(self_loop_positions)})'))

        ax.legend(handles=legend_elements, loc='upper right')

# Convenience function for single trace plotting
def plot_hmm_segmentation_and_path(
        signal: np.ndarray,
        segment_results: Dict[str, Any],
        state_sequence: List[str],
        full_path: List[str] = None,
        signal_key: str = "",
        save_path: Optional[str] = None,
        custom_colors: Optional[List[str]] = None
) -> None:
    """Convenience function for single trace plotting."""
    if full_path is None:
        full_path = state_sequence

    visualizer = HMMVisualizer(custom_colors=custom_colors)
    visualizer.plot_hmm_segmentation(
        signal=signal,
        segment_results=segment_results,
        state_sequence=state_sequence,
        full_path=full_path,
        signal_key=signal_key,
        save_path=save_path
    )

def plot_segmentation_only(
        signal: np.ndarray,
        segment_results: Dict[str, Any],
        signal_key: str = "",
        save_path: Optional[str] = None,
        custom_colors: Optional[List[str]] = None
) -> None:
    """Plot only segmented signal without HMM inference."""
    breakpoints = segment_results['breakpoints']
    means = segment_results['means']
    visualizer = HMMVisualizer(custom_colors=custom_colors)
    fig, ax = plt.subplots(figsize=(15, 5))
    visualizer._plot_segmented_signal(ax, signal, breakpoints, means, signal_key, segment_results)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_multi_panel_hmm_states(
        results_dict: Dict[str, Dict[str, Any]],
        max_panels: int = 10,
        save_path: Optional[str] = None,
        custom_colors: Optional[List[str]] = None,
        sort_by: str = 'log_probability',
        figsize: Optional[Tuple[int, int]] = None,
        title: Optional[str] = None
) -> None:
    """Create multi-panel plot with barycenter profile bands."""
    visualizer = HMMVisualizer(custom_colors=custom_colors)

    # Sort traces
    if sort_by == 'log_probability':
        sorted_keys = sorted(results_dict.keys(),
                             key=lambda k: results_dict[k].get('log_probability', float('-inf')),
                             reverse=True)
    elif sort_by == 'reverse_log_probability':
        sorted_keys = sorted(results_dict.keys(),
                             key=lambda k: results_dict[k].get('log_probability', float('-inf')))
    else:
        sorted_keys = list(results_dict.keys())

    selected_keys = sorted_keys[:min(max_panels, len(sorted_keys))]
    num_panels = len(selected_keys)

    if num_panels == 0:
        logger.warning("No results to plot")
        return

    # Calculate global limits
    global_x_max = 0
    all_normalized_signals = []

    for signal_key in selected_keys:
        result = results_dict[signal_key]
        normalized_signal = result['z_normalized_signal']
        all_normalized_signals.append(normalized_signal)
        global_x_max = max(global_x_max, len(normalized_signal))

    all_values = np.concatenate(all_normalized_signals)
    global_y_min = np.min(all_values)
    global_y_max = np.max(all_values)
    y_range = global_y_max - global_y_min
    global_y_min -= 0.15 * y_range
    global_y_max += 0.05 * y_range

    # Create figure
    if figsize is None:
        figsize = (7, max(5, num_panels * 0.8))

    fig, axes = plt.subplots(num_panels, 1, figsize=figsize, sharex=True, sharey=True)
    if num_panels == 1:
        axes = [axes]

    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold', y=0.995)

    # Plot each trace
    for idx, (ax, signal_key) in enumerate(zip(axes, selected_keys)):
        result = results_dict[signal_key]
        segment_results = result['segment_results']
        full_path = result.get('full_path', [])
        log_prob = result.get('log_probability', float('-inf'))
        amino_acid = result.get('amino_acid', 'unknown')
        breakpoints = segment_results['breakpoints']
        num_segments = len(breakpoints) - 1
        normalized_signal = result['z_normalized_signal']

        channel = extract_channel_from_key(signal_key)

        # Map segments to states
        match_states_in_path = [s for s in full_path if 'Match' in s]
        segment_to_match = {}
        unique_match_states = set()
        for obs_idx in range(min(len(match_states_in_path), num_segments)):
            segment_to_match[obs_idx] = match_states_in_path[obs_idx]
            match_pos = int(match_states_in_path[obs_idx].split('_')[1])
            unique_match_states.add(match_pos)

        num_unique = len(unique_match_states)

        subplot_title = f"Ch{channel} | AA: {amino_acid} | LogP: {log_prob:.2f} | {num_unique} unique matches"
        ax.set_title(subplot_title, fontsize=9, loc='left', pad=2)
        ax.set_ylabel("Z-Norm", fontsize=9)

        # Find boundaries
        skip_boundaries, slip_boundaries, self_loop_boundaries = [], [], []
        prev_state, prev_match_idx, obs_count = None, -1, 0

        for state in full_path:
            if 'Match' in state:
                curr_match_idx = int(state.split('_')[1])
                if prev_match_idx >= 0:
                    if curr_match_idx > prev_match_idx + 1 and obs_count < len(breakpoints):
                        skip_boundaries.append(breakpoints[obs_count])
                    if curr_match_idx < prev_match_idx and obs_count < len(breakpoints):
                        slip_boundaries.append(breakpoints[obs_count])
                if state == prev_state and 0 < obs_count < len(breakpoints):
                    self_loop_boundaries.append(breakpoints[obs_count])
                prev_match_idx = curr_match_idx
                obs_count += 1
                prev_state = state

        # Plot segments
        for seg_idx in range(num_segments):
            start_idx = breakpoints[seg_idx]
            end_idx = breakpoints[seg_idx + 1]
            segment_range = np.arange(start_idx, end_idx)

            if seg_idx in segment_to_match:
                match_state = segment_to_match[seg_idx]
                match_position = int(match_state.split('_')[1])
                color, alpha_val = visualizer.get_segment_color(match_position, num_segments)

                # Use match position for profile lookup
                if 'hmm_profile_stats' in segment_results and str(match_position) in segment_results[
                    'hmm_profile_stats']:
                    profile_mean = segment_results['hmm_profile_stats'][str(match_position)][0]
                    profile_std = segment_results['hmm_profile_stats'][str(match_position)][1]

                    ax.fill_between(
                        segment_range,
                        profile_mean - profile_std,
                        profile_mean + profile_std,
                        color=color,
                        alpha=alpha_val * 0.2,
                        linewidth=0,
                        zorder=5
                    )

                    ax.hlines(profile_mean, start_idx, end_idx,
                              colors=color, linewidth=1.2, alpha=alpha_val * 0.7, zorder=6)

                # Plot signal
                ax.plot(segment_range, normalized_signal[start_idx:end_idx],
                        color=color, alpha=alpha_val, linewidth=0.75, zorder=7)

                # Add match state label
                label_y = global_y_min + 0.04 * (global_y_max - global_y_min)
                mid_point = (start_idx + end_idx) / 2
                ax.text(mid_point, label_y, f'{match_position}',
                        ha='center', va='bottom', fontsize=7,
                        color=color, weight='bold', alpha=min(alpha_val + 0.3, 1.0), zorder=8)

        # Add boundaries
        for skip_pos in skip_boundaries:
            ax.axvline(x=skip_pos, color='red', linestyle='--', linewidth=1.5, alpha=0.9, zorder=15)
        for slip_pos in slip_boundaries:
            ax.axvline(x=slip_pos, color='blue', linestyle='--', linewidth=1.5, alpha=0.9, zorder=15)
        for self_loop_pos in self_loop_boundaries:
            ax.axvline(x=self_loop_pos, color='magenta', linestyle=':', linewidth=2.0, alpha=0.8, zorder=15)

        # State count annotation
        match_count = sum(1 for s in full_path if 'Match' in s)
        skip_count = sum(1 for s in full_path if 'Skip' in s)
        slip_count = sum(1 for s in full_path if 'Slip' in s)
        stats_text = f"M:{match_count} K:{skip_count} L:{slip_count}"
        ax.text(0.98, 0.95, stats_text, transform=ax.transAxes,
                fontsize=8, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        ax.set_xlim(0, global_x_max)
        ax.set_ylim(global_y_min, global_y_max)
        ax.tick_params(axis='both', labelsize=8)
        ax.grid(True, axis='y', alpha=0.15, linewidth=0.3)

        # Force spines
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_linewidth(2.0)
        ax.spines['left'].set_linewidth(2.0)

    axes[-1].set_xlabel("Sample Index", fontsize=9)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_segment_pileup(
        results_dict: Dict[str, Dict[str, Any]],
        max_traces: int = 10,
        save_path: Optional[str] = None,
        custom_colors: Optional[List[str]] = None,
        amino_acid: Optional[str] = None,
        segment_width: int = 80
) -> None:
    """Create segment pileup plot."""
    if custom_colors is None:
        custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Filter by amino acid if specified
    if amino_acid:
        filtered_results = {k: v for k, v in results_dict.items()
                            if v.get('amino_acid') == amino_acid}
    else:
        filtered_results = results_dict

    # Sort and select traces
    sorted_keys = sorted(filtered_results.keys(),
                         key=lambda k: filtered_results[k].get('log_probability', float('-inf')),
                         reverse=True)
    selected_keys = sorted_keys[:min(max_traces, len(sorted_keys))]
    num_traces = len(selected_keys)

    if num_traces == 0:
        logger.warning("No results to plot")
        return

    fig, ax = plt.subplots(1, 1, figsize=(7, 2.25))

    title = f"Normalized Segment Profile Pileup - {amino_acid} ({num_traces} traces)" if amino_acid else f"Profile Pileup ({num_traces} traces)"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel("Z-normalized Mean Current")
    ax.grid(True, alpha=0.2, linewidth=0.5)

    all_trace_profiles = []
    trace_info = []
    num_segments = None

    for trace_idx, signal_key in enumerate(selected_keys):
        result = filtered_results[signal_key]
        segment_results = result['segment_results']
        channel = extract_channel_from_key(signal_key)

        if 'z_normalized_stats' in segment_results:
            z_means = [segment_results['z_normalized_stats'][str(i)][0]
                       for i in range(len(segment_results['means']))]
        else:
            means_array = np.array(segment_results['means'])
            z_means = (means_array - np.mean(means_array)) / np.std(means_array, ddof=1) if np.std(
                means_array) > 0 else means_array - np.mean(means_array)

        profile = []
        for mean_val in z_means:
            profile.extend([mean_val] * segment_width)

        all_trace_profiles.append(np.array(profile))

        if num_segments is None:
            num_segments = len(z_means)

        trace_info.append({
            'channel': channel,
            'log_prob': result.get('log_probability', float('-inf'))
        })

    total_length = num_segments * segment_width
    x_positions = np.arange(total_length)

    # Plot traces
    for trace_idx, (profile, info) in enumerate(zip(all_trace_profiles, trace_info)):
        color = custom_colors[trace_idx % len(custom_colors)]
        label = f"Ch{info['channel']} (LogP:{info['log_prob']:.1f})"
        ax.plot(x_positions, profile,
                color=color, alpha=0.7, linewidth=1.5,
                label=label, drawstyle='steps-mid')

    # Add segment boundaries
    for seg_idx in range(num_segments + 1):
        x_pos = seg_idx * segment_width
        if seg_idx < num_segments:
            ax.axvline(x=x_pos, color='gray', linestyle='--', linewidth=0.5, alpha=0.4)
            mid_point = x_pos + segment_width // 2
            ax.text(mid_point, ax.get_ylim()[0], str(seg_idx),
                    ha='center', va='top', fontsize=9, color='black', alpha=0.65)

    # Add mean profile
    if num_traces > 1:
        mean_profile = np.mean(np.array(all_trace_profiles), axis=0)
        ax.plot(x_positions, mean_profile,
                color='black', linewidth=3, alpha=0.95,
                label='Mean Profile', zorder=10, drawstyle='steps-mid')

    ax.axhline(y=0, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xlim(0, total_length)
    ax.set_ylim(-3.0, 3.0)

    # Set x-axis labels
    segment_ticks = [i * segment_width for i in range(0, num_segments + 1, 5)]
    segment_labels = [f"{i}" for i in range(0, num_segments + 1, 5)]
    ax.set_xticks(segment_ticks)
    ax.set_xticklabels(segment_labels)
    ax.set_xlabel("Segment Index")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.close()

def plot_match_state_pileup(
        results_dict: Dict[str, Dict[str, Any]],
        amino_acid: Optional[str] = None,
        barycenter_profile_stats: Optional[Dict[str, Tuple[float, float]]] = None,
        max_traces: Optional[int] = None,
        save_path: Optional[str] = None,
        custom_colors: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (8, 5.5),
        title: Optional[str] = None,
        alpha_per_trace: float = 0.3
) -> None:
    """Create pileup plot showing segment means grouped by MATCH STATE they aligned to."""
    if custom_colors is None:
        custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Filter to specified amino acid
    if amino_acid:
        filtered_results = {k: v for k, v in results_dict.items()
                            if v.get('amino_acid') == amino_acid}
    else:
        filtered_results = results_dict

    if not filtered_results:
        logger.warning(f"No results found" + (f" for amino acid {amino_acid}" if amino_acid else ""))
        return

    # Limit traces if specified
    if max_traces:
        sorted_keys = sorted(filtered_results.keys(),
                             key=lambda k: filtered_results[k].get('log_probability', float('-inf')),
                             reverse=True)
        selected_keys = sorted_keys[:max_traces]
        filtered_results = {k: filtered_results[k] for k in selected_keys}

    # Collect segments by match state
    match_state_data = {i: [] for i in range(35)}

    for signal_key, result in filtered_results.items():
        segment_results = result['segment_results']
        full_path = result.get('full_path', [])

        match_states_in_path = [s for s in full_path if 'Match' in s]

        if 'z_normalized_stats' in segment_results:
            for obs_idx, match_state in enumerate(match_states_in_path):
                if obs_idx < len(segment_results['means']):
                    match_position = int(match_state.split('_')[1])
                    z_mean = segment_results['z_normalized_stats'][str(obs_idx)][0]

                    match_state_data[match_position].append({
                        'z_mean': z_mean,
                        'trace': signal_key,
                        'log_prob': result.get('log_probability', float('-inf'))
                    })

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    if title is None:
        if amino_acid:
            title = f"Match State Pileup - {amino_acid} ({len(filtered_results)} traces)"
        else:
            title = f"Match State Pileup - All Amino Acids ({len(filtered_results)} traces)"

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Match State Position", fontsize=12, fontweight='bold')
    ax.set_ylabel("Z-Normalized Mean Current", fontsize=12, fontweight='bold')

    # Plot each match state's data
    x_offset = 0.4
    bar_width = 0.5

    for match_pos in range(35):
        data_points = match_state_data[match_pos]
        color = custom_colors[match_pos % len(custom_colors)]

        # Plot profile band (expected from barycenter)
        if barycenter_profile_stats and str(match_pos) in barycenter_profile_stats:
            profile_mean = barycenter_profile_stats[str(match_pos)][0]
            profile_std = barycenter_profile_stats[str(match_pos)][1]

            ax.fill_between(
                [match_pos - bar_width / 2, match_pos + bar_width / 2],
                [profile_mean - profile_std, profile_mean - profile_std],
                [profile_mean + profile_std, profile_mean + profile_std],
                color=color,
                alpha=0.2,
                zorder=1,
                linewidth=0
            )

            ax.hlines(profile_mean, match_pos - bar_width / 2, match_pos + bar_width / 2,
                      colors=color, linewidth=2, alpha=0.5, zorder=2,
                      linestyle='--', label='Profile' if match_pos == 0 else '')

        # Plot observed data points
        if data_points:
            z_means = [d['z_mean'] for d in data_points]

            n_points = len(z_means)
            if n_points == 1:
                x_positions = [match_pos]
            else:
                jitter = np.linspace(-x_offset, x_offset, n_points)
                x_positions = match_pos + jitter

            ax.scatter(x_positions, z_means,
                       alpha=alpha_per_trace,
                       s=20,
                       color=color,
                       edgecolors='none',
                       zorder=5,
                       label='Observed' if match_pos == 0 else '')

            observed_mean = np.mean(z_means)
            ax.hlines(observed_mean, match_pos - x_offset, match_pos + x_offset,
                      colors=color, linewidth=3, alpha=0.9, zorder=10,
                      label='Observed Mean' if match_pos == 0 else '')

            ax.text(match_pos, ax.get_ylim()[1] * 0.95, f'{n_points}',
                    ha='center', va='top', fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7),
                    zorder=15)

    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, zorder=3)
    ax.set_xlim(-1, 35)
    ax.set_ylim(-3, 3)
    ax.set_xticks(range(0, 35, 5))
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.close()

def plot_backslip_distribution(
        results_dict: Dict[str, Dict[str, Any]],
        amino_acid: Optional[str] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
        title: Optional[str] = None,
        color: str = '#1f77b4',
        show_stats: bool = True
) -> None:
    """Plot distribution of backslip sizes."""
    backslip_counts = defaultdict(int)

    if amino_acid:
        filtered_results = {k: v for k, v in results_dict.items()
                            if v.get('amino_acid') == amino_acid}
    else:
        filtered_results = results_dict

    for signal_key, result in filtered_results.items():
        full_path = result.get('full_path', [])
        if not full_path:
            continue

        prev_match_idx = -1
        for state in full_path:
            if 'Match' in state:
                try:
                    curr_match_idx = int(state.split('_')[1])
                    if prev_match_idx >= 0 and curr_match_idx < prev_match_idx:
                        backslip_size = prev_match_idx - curr_match_idx
                        backslip_counts[backslip_size] += 1
                    prev_match_idx = curr_match_idx
                except (ValueError, IndexError):
                    continue

    if not backslip_counts:
        logger.info(f"No backslips found" + (f" for amino acid {amino_acid}" if amino_acid else ""))
        return

    max_backslip = max(backslip_counts.keys())
    x_values = list(range(1, max_backslip + 1))
    y_values = [backslip_counts.get(i, 0) for i in x_values]

    fig, ax = plt.subplots(figsize=figsize)

    if title is None:
        title = f"Backslip Distribution - {amino_acid}" if amino_acid else "Backslip Distribution - All AAs"

    ax.set_title(title, fontsize=14, fontweight='bold')
    bars = ax.bar(x_values, y_values, color=color, alpha=0.7, edgecolor='black', linewidth=1)

    for bar, value in zip(bars, y_values):
        if value > 0:
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                    f'{int(value)}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel("Backslip Size (Match State Positions)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Frequency", fontsize=12, fontweight='bold')
    ax.set_xticks(x_values)
    ax.grid(True, axis='y', alpha=0.3)

    if show_stats and backslip_counts:
        total_backslips = sum(backslip_counts.values())
        total_signals = len(filtered_results)
        avg_size = sum(s * c for s, c in backslip_counts.items()) / total_backslips

        stats_text = f"Total Backslips: {total_backslips}\n"
        stats_text += f"Signals Analyzed: {total_signals}\n"
        stats_text += f"Avg Backslip Size: {avg_size:.2f}\n"
        stats_text += f"Max Backslip Size: {max(backslip_counts.keys())}"

        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=10, ha='right', va='top', bbox=props)

    ax.set_ylim(bottom=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.close()

def plot_skip_distribution(
        results_dict: Dict[str, Dict[str, Any]],
        amino_acid: Optional[str] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
        title: Optional[str] = None,
        color: str = '#ff7f0e',
        show_stats: bool = True
) -> None:
    """Plot distribution of skip sizes."""
    skip_counts = defaultdict(int)

    if amino_acid:
        filtered_results = {k: v for k, v in results_dict.items()
                            if v.get('amino_acid') == amino_acid}
    else:
        filtered_results = results_dict

    for signal_key, result in filtered_results.items():
        full_path = result.get('full_path', [])
        if not full_path:
            continue

        prev_match_idx = -1
        for state in full_path:
            if 'Match' in state:
                try:
                    curr_match_idx = int(state.split('_')[1])
                    if prev_match_idx >= 0 and curr_match_idx > prev_match_idx + 1:
                        skip_size = curr_match_idx - prev_match_idx - 1
                        skip_counts[skip_size] += 1
                    prev_match_idx = curr_match_idx
                except (ValueError, IndexError):
                    continue

    if not skip_counts:
        logger.info(f"No skips found" + (f" for amino acid {amino_acid}" if amino_acid else ""))
        return

    max_skip = max(skip_counts.keys())
    x_values = list(range(1, max_skip + 1))
    y_values = [skip_counts.get(i, 0) for i in x_values]

    fig, ax = plt.subplots(figsize=figsize)

    if title is None:
        title = f"Skip Distribution - {amino_acid}" if amino_acid else "Skip Distribution - All AAs"

    ax.set_title(title, fontsize=14, fontweight='bold')
    bars = ax.bar(x_values, y_values, color=color, alpha=0.7, edgecolor='black', linewidth=1)

    for bar, value in zip(bars, y_values):
        if value > 0:
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                    f'{int(value)}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel("Skip Size (Match State Positions)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Frequency", fontsize=12, fontweight='bold')
    ax.set_xticks(x_values)
    ax.grid(True, axis='y', alpha=0.3)

    if show_stats and skip_counts:
        total_skips = sum(skip_counts.values())
        total_signals = len(filtered_results)
        avg_size = sum(s * c for s, c in skip_counts.items()) / total_skips

        stats_text = f"Total Skips: {total_skips}\n"
        stats_text += f"Signals Analyzed: {total_signals}\n"
        stats_text += f"Avg Skip Size: {avg_size:.2f}\n"
        stats_text += f"Max Skip Size: {max(skip_counts.keys())}"

        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=10, ha='right', va='top', bbox=props)

    ax.set_ylim(bottom=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.close()

def plot_backslip_by_position(
        results_dict: Dict[str, Dict[str, Any]],
        amino_acid: Optional[str] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 6),
        title: Optional[str] = None,
        show_stats: bool = True
) -> None:
    """Plot frequency of backslips and skips by match state position."""
    backslip_from_position = defaultdict(int)
    backslip_to_position = defaultdict(int)
    skip_from_position = defaultdict(int)

    if amino_acid:
        filtered_results = {k: v for k, v in results_dict.items()
                            if v.get('amino_acid') == amino_acid}
    else:
        filtered_results = results_dict

    total_backslips = 0
    total_skips = 0

    for signal_key, result in filtered_results.items():
        full_path = result.get('full_path', [])
        if not full_path:
            continue

        prev_match_idx = -1
        for state in full_path:
            if 'Match' in state:
                try:
                    curr_match_idx = int(state.split('_')[1])
                    if prev_match_idx >= 0:
                        if curr_match_idx < prev_match_idx:
                            backslip_from_position[prev_match_idx] += 1
                            backslip_to_position[curr_match_idx] += 1
                            total_backslips += 1
                        elif curr_match_idx > prev_match_idx + 1:
                            skip_from_position[prev_match_idx] += 1
                            total_skips += 1
                    prev_match_idx = curr_match_idx
                except (ValueError, IndexError):
                    continue

    # Check if there's any data to plot
    if total_backslips == 0 and total_skips == 0:
        logger.info(f"No backslips or skips found for plotting" +
                   (f" for amino acid {amino_acid}" if amino_acid else ""))
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    if title is None:
        if amino_acid:
            title = f"Backslip & Skip Frequency by Position - {amino_acid}"
        else:
            title = "Backslip & Skip Frequency by Position - All AAs"

    fig.suptitle(title, fontsize=14, fontweight='bold')

    x_positions = list(range(35))

    # Top subplot: Backslips
    backslip_values = [backslip_from_position.get(i, 0) for i in x_positions]
    bars1 = ax1.bar(x_positions, backslip_values, color='#d62728', alpha=0.7,
                    edgecolor='black', linewidth=0.5, label='Backslips FROM')

    for bar, value in zip(bars1, backslip_values):
        if value > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                     f'{int(value)}', ha='center', va='bottom', fontsize=8)

    ax1.set_ylabel("Backslip Frequency", fontsize=11, fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Add secondary axis for backslips TO
    backslip_to_values = [backslip_to_position.get(i, 0) for i in x_positions]
    ax1_twin = ax1.twinx()
    ax1_twin.plot(x_positions, backslip_to_values, 'o-', color='#ff7f0e',
                  alpha=0.6, linewidth=2, markersize=4, label='Backslips TO')
    ax1_twin.set_ylabel("Backslip TO Frequency", fontsize=11, fontweight='bold', color='#ff7f0e')
    ax1_twin.tick_params(axis='y', labelcolor='#ff7f0e')
    ax1_twin.set_ylim(bottom=0)

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

    # Bottom subplot: Skips
    skip_values = [skip_from_position.get(i, 0) for i in x_positions]
    bars2 = ax2.bar(x_positions, skip_values, color='#2ca02c', alpha=0.7,
                    edgecolor='black', linewidth=0.5)

    for bar, value in zip(bars2, skip_values):
        if value > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                     f'{int(value)}', ha='center', va='bottom', fontsize=8)

    ax2.set_xlabel("Match State Position", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Skip Frequency", fontsize=11, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.set_ylim(bottom=0)
    ax2.set_xticks(range(0, 35, 5))
    ax2.set_xlim(-0.5, 34.5)

    # Add statistics
    if show_stats and (total_backslips > 0 or total_skips > 0):
        stats_text = f"Total Backslips: {total_backslips}\n"
        if backslip_from_position:
            max_pos = max(backslip_from_position.keys(), key=lambda k: backslip_from_position[k])
            stats_text += f"Most Backslips FROM: Match_{max_pos} ({backslip_from_position[max_pos]})\n"
        stats_text += f"Total Skips: {total_skips}"
        if skip_from_position:
            max_skip_pos = max(skip_from_position.keys(), key=lambda k: skip_from_position[k])
            stats_text += f"\nMost Skips FROM: Match_{max_skip_pos} ({skip_from_position[max_skip_pos]})"

        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                 fontsize=9, ha='left', va='top', bbox=props)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.close()