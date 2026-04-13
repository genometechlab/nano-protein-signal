"""Signal visualization utilities for HMM analysis."""

import logging
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)

DEFAULT_MATCH_STATES = 35

DEFAULT_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]


def configure_logging():
    """Suppress verbose logging from matplotlib, fontTools, and PIL."""
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    warnings.filterwarnings("ignore", category=FutureWarning, module="matplotlib")

    for name in ['matplotlib', 'matplotlib.font_manager', 'matplotlib.backends',
                 'fontTools', 'fontTools.subset', 'PIL', 'PIL.PngImagePlugin']:
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.getLogger('fontTools.subset').setLevel(logging.ERROR)


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


# ── Path analysis helpers ─────────────────────────────────────────────

def _extract_match_indices(full_path: List[str]) -> List[int]:
    """Extract ordered match state indices from a Viterbi path."""
    indices = []
    for state in full_path:
        if 'Match' in state:
            try:
                indices.append(int(state.split('_')[1]))
            except (ValueError, IndexError):
                continue
    return indices


def _detect_boundaries(full_path: List[str], breakpoints: List[int]) -> Dict[str, List[int]]:
    """Detect skip, slip, self-loop, and insert boundaries from a path."""
    skip_bounds = []
    slip_bounds = []
    self_loop_bounds = []
    insert_bounds = []

    prev_state = None
    prev_match_idx = -1
    obs_count = 0

    for state in full_path:
        if 'Match' in state:
            curr_idx = int(state.split('_')[1])

            if prev_match_idx >= 0:
                if curr_idx > prev_match_idx + 1 and obs_count < len(breakpoints):
                    skip_bounds.append(breakpoints[obs_count])
                if curr_idx < prev_match_idx and obs_count < len(breakpoints):
                    slip_bounds.append(breakpoints[obs_count])

            if state == prev_state and 0 < obs_count < len(breakpoints):
                self_loop_bounds.append(breakpoints[obs_count])

            prev_match_idx = curr_idx
            obs_count += 1
            prev_state = state

        elif 'Insert' in state:
            if obs_count < len(breakpoints):
                insert_bounds.append(breakpoints[obs_count])
            obs_count += 1

    return {
        'skip': skip_bounds,
        'slip': slip_bounds,
        'self_loop': self_loop_bounds,
        'insert': insert_bounds,
    }


def _collect_path_events(
    full_path: List[str],
    event_type: str
) -> Dict[int, int]:
    """Count backslip or skip events by size from a Viterbi path.

    event_type: 'backslip' or 'skip'
    """
    counts: Dict[int, int] = defaultdict(int)
    prev_match_idx = -1

    for state in full_path:
        if 'Match' not in state:
            continue
        try:
            curr = int(state.split('_')[1])
        except (ValueError, IndexError):
            continue

        if prev_match_idx >= 0:
            if event_type == 'backslip' and curr < prev_match_idx:
                counts[prev_match_idx - curr] += 1
            elif event_type == 'skip' and curr > prev_match_idx + 1:
                counts[curr - prev_match_idx - 1] += 1

        prev_match_idx = curr

    return counts


# ── Shared utilities ──────────────────────────────────────────────────

def extract_channel_from_key(signal_key: str) -> str:
    parts = signal_key.split('_')
    return parts[1] if len(parts) >= 2 else "unknown"


def _filter_results(
    results_dict: Dict[str, Dict[str, Any]],
    amino_acid: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    if amino_acid:
        return {k: v for k, v in results_dict.items() if v.get('amino_acid') == amino_acid}
    return results_dict


def _configure_spines(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['left'].set_linewidth(2.0)


# ── HMMVisualizer ─────────────────────────────────────────────────────

class HMMVisualizer:
    """Visualizer for HMM segmentation results with barycenter profile bands."""

    def __init__(self, custom_colors: Optional[List[str]] = None):
        if 'genometechlab_main' in plt.style.available:
            plt.style.use('genometechlab_main')

        mpl.rcParams['figure.autolayout'] = False
        mpl.rcParams['figure.constrained_layout.use'] = False

        self.color_blue = '#1f77b4'
        self.color_orange = '#ff7f0e'
        self.custom_colors = custom_colors or DEFAULT_COLORS
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
        """Plot HMM segmentation: top panel = raw segments, bottom = HMM-colored."""
        breakpoints = segment_results['breakpoints']
        means = segment_results['means']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

        self._plot_segmented_signal(ax1, signal, breakpoints, means, signal_key)
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
        """Plot segmented signal without HMM inference — sequential coloring."""
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

            mid_point = (start_idx + end_idx) / 2
            y_limits = ax.get_ylim()
            label_y = (y_limits[0] + 0.03 * (y_limits[1] - y_limits[0])
                       if y_limits != (0, 1) else -2.8)

            ax.text(mid_point, label_y, str(i),
                    ha='center', va='bottom', fontsize=7,
                    color=color, weight='bold', alpha=min(alpha + 0.3, 1.0), zorder=8)

        ax.set_xlim(0, len(signal))
        _configure_spines(ax)

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
        """Plot HMM state-colored signal with profile bands by match state."""
        channel = extract_channel_from_key(title_suffix)
        ax.set_title(f"HMM State-Colored Signal - Channel {channel}")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Z-Normalized Current")

        segment_to_state = {}
        obs_idx = 0
        for state in full_path:
            if 'Match' in state or 'Insert' in state:
                if obs_idx < len(breakpoints) - 1:
                    kind = 'match' if 'Match' in state else 'insert'
                    segment_to_state[obs_idx] = (kind, state)
                    obs_idx += 1

        boundaries = _detect_boundaries(full_path, breakpoints)

        num_segments = len(breakpoints) - 1
        unique_match_states = set()
        for state_type, state_name in segment_to_state.values():
            if state_type == 'match':
                unique_match_states.add(int(state_name.split('_')[1]))

        for seg_idx in range(num_segments):
            start_idx = breakpoints[seg_idx]
            end_idx = breakpoints[seg_idx + 1]
            segment_range = np.arange(start_idx, end_idx)

            if seg_idx not in segment_to_state:
                ax.plot(segment_range, signal[start_idx:end_idx],
                        color=self.skip_color, alpha=0.3, linewidth=0.5, zorder=7)
                continue

            state_type, state_name = segment_to_state[seg_idx]

            if state_type == 'insert':
                self._plot_insert_segment(ax, signal, segment_range, start_idx,
                                          end_idx, state_name)
            elif state_type == 'match':
                match_pos = int(state_name.split('_')[1])
                self._plot_match_segment(ax, signal, segment_range, start_idx,
                                         end_idx, match_pos, num_segments, segment_results)

        self._draw_boundary_lines(ax, boundaries)
        ax.set_xlim(0, len(signal))
        _configure_spines(ax)
        self._add_hmm_legend(ax, len(unique_match_states), boundaries)

    def _plot_insert_segment(self, ax, signal, segment_range, start_idx, end_idx, state_name):
        insert_color = '#00ff00'
        ax.plot(segment_range, signal[start_idx:end_idx],
                color=insert_color, alpha=0.7, linewidth=0.75, zorder=8)

        parts = state_name.split('_')
        label = f'i_{parts[1]}_{parts[2]}' if len(parts) >= 3 else 'ins'

        mid_point = (start_idx + end_idx) / 2
        y_limits = ax.get_ylim()
        label_y = y_limits[0] + 0.03 * (y_limits[1] - y_limits[0])
        ax.text(mid_point, label_y, label,
                ha='center', va='bottom', fontsize=8,
                color=insert_color, weight='bold', style='italic', zorder=9)

    def _plot_match_segment(self, ax, signal, segment_range, start_idx, end_idx,
                            match_pos, num_segments, segment_results):
        color, alpha_val = self.get_segment_color(match_pos, num_segments)

        if (segment_results and 'hmm_profile_stats' in segment_results
                and str(match_pos) in segment_results['hmm_profile_stats']):
            profile_mean, profile_std = segment_results['hmm_profile_stats'][str(match_pos)]

            ax.fill_between(segment_range,
                            profile_mean - profile_std, profile_mean + profile_std,
                            color=color, alpha=alpha_val * 0.3, linewidth=0, zorder=5)
            ax.hlines(profile_mean, start_idx, end_idx,
                      colors=color, linewidth=1.0, alpha=alpha_val * 0.8, zorder=6)

        ax.plot(segment_range, signal[start_idx:end_idx],
                color=color, alpha=alpha_val, linewidth=0.50, zorder=7)

        mid_point = (start_idx + end_idx) / 2
        y_limits = ax.get_ylim()
        label_y = y_limits[0] + 0.03 * (y_limits[1] - y_limits[0])
        ax.text(mid_point, label_y, f'{match_pos}',
                ha='center', va='bottom', fontsize=10,
                color=color, weight='bold', alpha=min(alpha_val + 0.3, 1.0), zorder=8)

    @staticmethod
    def _draw_boundary_lines(ax, boundaries):
        styles = {
            'insert': ('green', ':', 1.5, 0.8),
            'skip': ('red', '--', 0.75, 0.9),
            'slip': ('blue', '--', 0.75, 0.9),
            'self_loop': ('magenta', ':', 0.75, 0.8),
        }
        for kind, (color, ls, lw, alpha) in styles.items():
            for pos in boundaries.get(kind, []):
                ax.axvline(x=pos, color=color, linestyle=ls, linewidth=lw, alpha=alpha, zorder=15)

    @staticmethod
    def _add_hmm_legend(ax, num_unique_matches, boundaries):
        elements = [
            mpatches.Patch(color='#1f77b4', alpha=0.6,
                           label=f'Match States: {num_unique_matches} unique')
        ]
        if boundaries['insert']:
            elements.append(mpatches.Patch(color='#00ff00', alpha=0.7,
                                           label=f'Insert States ({len(boundaries["insert"])})'))
            elements.append(plt.Line2D([0], [0], color='green', linestyle=':', linewidth=1.5,
                                       label='Insert Boundaries'))
        if boundaries['skip']:
            elements.append(plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1.5,
                                       label=f'Skip Boundaries ({len(boundaries["skip"])})'))
        if boundaries['slip']:
            elements.append(plt.Line2D([0], [0], color='blue', linestyle='--', linewidth=1.5,
                                       label=f'Slip Boundaries ({len(boundaries["slip"])})'))
        if boundaries['self_loop']:
            elements.append(plt.Line2D([0], [0], color='magenta', linestyle=':', linewidth=2.0,
                                       label=f'Self-loops ({len(boundaries["self_loop"])})'))
        ax.legend(handles=elements, loc='upper right')


# ── Convenience functions ─────────────────────────────────────────────

def plot_hmm_segmentation_and_path(
    signal: np.ndarray,
    segment_results: Dict[str, Any],
    state_sequence: List[str],
    full_path: Optional[List[str]] = None,
    signal_key: str = "",
    save_path: Optional[str] = None,
    custom_colors: Optional[List[str]] = None
) -> None:
    """Convenience wrapper for single trace plotting."""
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

    if sort_by == 'log_probability':
        sorted_keys = sorted(results_dict, key=lambda k: results_dict[k].get('log_probability', -np.inf), reverse=True)
    elif sort_by == 'reverse_log_probability':
        sorted_keys = sorted(results_dict, key=lambda k: results_dict[k].get('log_probability', -np.inf))
    else:
        sorted_keys = list(results_dict.keys())

    selected_keys = sorted_keys[:min(max_panels, len(sorted_keys))]
    num_panels = len(selected_keys)

    if num_panels == 0:
        logger.warning("No results to plot")
        return

    all_signals = [results_dict[k]['z_normalized_signal'] for k in selected_keys]
    global_x_max = max(len(s) for s in all_signals)
    all_values = np.concatenate(all_signals)
    y_range = np.ptp(all_values)
    global_y_min = np.min(all_values) - 0.15 * y_range
    global_y_max = np.max(all_values) + 0.05 * y_range

    if figsize is None:
        figsize = (7, max(5, num_panels * 0.8))

    fig, axes = plt.subplots(num_panels, 1, figsize=figsize, sharex=True, sharey=True)
    if num_panels == 1:
        axes = [axes]

    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold', y=0.995)

    for idx, (ax, signal_key) in enumerate(zip(axes, selected_keys)):
        result = results_dict[signal_key]
        segment_results = result['segment_results']
        full_path = result.get('full_path', [])
        log_prob = result.get('log_probability', -np.inf)
        amino_acid = result.get('amino_acid', 'unknown')
        breakpoints = segment_results['breakpoints']
        num_segments = len(breakpoints) - 1
        normalized_signal = result['z_normalized_signal']
        channel = extract_channel_from_key(signal_key)

        match_states_in_path = [s for s in full_path if 'Match' in s]
        segment_to_match = {}
        unique_match_states = set()
        for obs_idx, ms in enumerate(match_states_in_path):
            if obs_idx < num_segments:
                segment_to_match[obs_idx] = ms
                unique_match_states.add(int(ms.split('_')[1]))

        ax.set_title(
            f"Ch{channel} | AA: {amino_acid} | LogP: {log_prob:.2f} | {len(unique_match_states)} unique matches",
            fontsize=9, loc='left', pad=2
        )
        ax.set_ylabel("Z-Norm", fontsize=9)

        boundaries = _detect_boundaries(full_path, breakpoints)

        for seg_idx in range(num_segments):
            start_idx = breakpoints[seg_idx]
            end_idx = breakpoints[seg_idx + 1]
            segment_range = np.arange(start_idx, end_idx)

            if seg_idx not in segment_to_match:
                continue

            match_state = segment_to_match[seg_idx]
            match_position = int(match_state.split('_')[1])
            color, alpha_val = visualizer.get_segment_color(match_position, num_segments)

            if ('hmm_profile_stats' in segment_results
                    and str(match_position) in segment_results['hmm_profile_stats']):
                profile_mean, profile_std = segment_results['hmm_profile_stats'][str(match_position)]
                ax.fill_between(segment_range,
                                profile_mean - profile_std, profile_mean + profile_std,
                                color=color, alpha=alpha_val * 0.2, linewidth=0, zorder=5)
                ax.hlines(profile_mean, start_idx, end_idx,
                          colors=color, linewidth=1.2, alpha=alpha_val * 0.7, zorder=6)

            ax.plot(segment_range, normalized_signal[start_idx:end_idx],
                    color=color, alpha=alpha_val, linewidth=0.75, zorder=7)

            label_y = global_y_min + 0.04 * (global_y_max - global_y_min)
            mid_point = (start_idx + end_idx) / 2
            ax.text(mid_point, label_y, f'{match_position}',
                    ha='center', va='bottom', fontsize=7,
                    color=color, weight='bold', alpha=min(alpha_val + 0.3, 1.0), zorder=8)

        for kind, color, ls, lw in [('skip', 'red', '--', 1.5), ('slip', 'blue', '--', 1.5),
                                     ('self_loop', 'magenta', ':', 2.0)]:
            for pos in boundaries[kind]:
                ax.axvline(x=pos, color=color, linestyle=ls, linewidth=lw, alpha=0.9, zorder=15)

        match_count = sum(1 for s in full_path if 'Match' in s)
        skip_count = sum(1 for s in full_path if 'Skip' in s)
        slip_count = sum(1 for s in full_path if 'Slip' in s)
        ax.text(0.98, 0.95, f"M:{match_count} K:{skip_count} L:{slip_count}",
                transform=ax.transAxes, fontsize=8, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        ax.set_xlim(0, global_x_max)
        ax.set_ylim(global_y_min, global_y_max)
        ax.tick_params(axis='both', labelsize=8)
        ax.grid(True, axis='y', alpha=0.15, linewidth=0.3)
        _configure_spines(ax)

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
    colors = custom_colors or DEFAULT_COLORS
    filtered = _filter_results(results_dict, amino_acid)

    sorted_keys = sorted(filtered, key=lambda k: filtered[k].get('log_probability', -np.inf), reverse=True)
    selected_keys = sorted_keys[:min(max_traces, len(sorted_keys))]
    num_traces = len(selected_keys)

    if num_traces == 0:
        logger.warning("No results to plot")
        return

    fig, ax = plt.subplots(1, 1, figsize=(7, 2.25))

    label = f"{amino_acid} ({num_traces} traces)" if amino_acid else f"({num_traces} traces)"
    ax.set_title(f"Normalized Segment Profile Pileup - {label}", fontsize=14, fontweight='bold')
    ax.set_ylabel("Z-normalized Mean Current")
    ax.grid(True, alpha=0.2, linewidth=0.5)

    all_profiles = []
    trace_info = []
    num_segments = None

    for signal_key in selected_keys:
        result = filtered[signal_key]
        segment_results = result['segment_results']
        channel = extract_channel_from_key(signal_key)

        if 'z_normalized_stats' in segment_results:
            z_means = [segment_results['z_normalized_stats'][str(i)][0]
                       for i in range(len(segment_results['means']))]
        else:
            means_array = np.array(segment_results['means'])
            std = np.std(means_array, ddof=1)
            z_means = ((means_array - np.mean(means_array)) / std if std > 0
                       else means_array - np.mean(means_array))

        profile = np.repeat(z_means, segment_width)
        all_profiles.append(profile)

        if num_segments is None:
            num_segments = len(z_means)

        trace_info.append({'channel': channel, 'log_prob': result.get('log_probability', -np.inf)})

    total_length = num_segments * segment_width
    x_positions = np.arange(total_length)

    for trace_idx, (profile, info) in enumerate(zip(all_profiles, trace_info)):
        color = colors[trace_idx % len(colors)]
        ax.plot(x_positions, profile, color=color, alpha=0.7, linewidth=1.5,
                label=f"Ch{info['channel']} (LogP:{info['log_prob']:.1f})", drawstyle='steps-mid')

    for seg_idx in range(num_segments):
        x_pos = seg_idx * segment_width
        ax.axvline(x=x_pos, color='gray', linestyle='--', linewidth=0.5, alpha=0.4)
        ax.text(x_pos + segment_width // 2, ax.get_ylim()[0], str(seg_idx),
                ha='center', va='top', fontsize=9, color='black', alpha=0.65)

    if num_traces > 1:
        mean_profile = np.mean(all_profiles, axis=0)
        ax.plot(x_positions, mean_profile, color='black', linewidth=3, alpha=0.95,
                label='Mean Profile', zorder=10, drawstyle='steps-mid')

    ax.axhline(y=0, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xlim(0, total_length)
    ax.set_ylim(-3.0, 3.0)

    segment_ticks = [i * segment_width for i in range(0, num_segments + 1, 5)]
    ax.set_xticks(segment_ticks)
    ax.set_xticklabels([str(i) for i in range(0, num_segments + 1, 5)])
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
    alpha_per_trace: float = 0.3,
    num_match_states: int = DEFAULT_MATCH_STATES
) -> None:
    """Create pileup plot showing segment means grouped by match state alignment."""
    colors = custom_colors or DEFAULT_COLORS
    filtered = _filter_results(results_dict, amino_acid)

    if not filtered:
        logger.warning(f"No results found" + (f" for amino acid {amino_acid}" if amino_acid else ""))
        return

    if max_traces:
        sorted_keys = sorted(filtered, key=lambda k: filtered[k].get('log_probability', -np.inf), reverse=True)
        filtered = {k: filtered[k] for k in sorted_keys[:max_traces]}

    match_state_data: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(num_match_states)}

    for signal_key, result in filtered.items():
        segment_results = result['segment_results']
        full_path = result.get('full_path', [])
        match_states_in_path = [s for s in full_path if 'Match' in s]

        if 'z_normalized_stats' in segment_results:
            for obs_idx, match_state in enumerate(match_states_in_path):
                if obs_idx < len(segment_results['means']):
                    match_pos = int(match_state.split('_')[1])
                    z_mean = segment_results['z_normalized_stats'][str(obs_idx)][0]
                    match_state_data[match_pos].append({
                        'z_mean': z_mean,
                        'trace': signal_key,
                        'log_prob': result.get('log_probability', -np.inf)
                    })

    fig, ax = plt.subplots(figsize=figsize)

    if title is None:
        label = amino_acid or "All Amino Acids"
        title = f"Match State Pileup - {label} ({len(filtered)} traces)"

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Match State Position", fontsize=12, fontweight='bold')
    ax.set_ylabel("Z-Normalized Mean Current", fontsize=12, fontweight='bold')

    x_offset = 0.4
    bar_width = 0.5

    for match_pos in range(num_match_states):
        data_points = match_state_data[match_pos]
        color = colors[match_pos % len(colors)]

        if barycenter_profile_stats and str(match_pos) in barycenter_profile_stats:
            profile_mean, profile_std = barycenter_profile_stats[str(match_pos)]

            ax.fill_between(
                [match_pos - bar_width / 2, match_pos + bar_width / 2],
                [profile_mean - profile_std] * 2,
                [profile_mean + profile_std] * 2,
                color=color, alpha=0.2, zorder=1, linewidth=0
            )
            ax.hlines(profile_mean, match_pos - bar_width / 2, match_pos + bar_width / 2,
                      colors=color, linewidth=2, alpha=0.5, zorder=2, linestyle='--',
                      label='Profile' if match_pos == 0 else '')

        if data_points:
            z_means = [d['z_mean'] for d in data_points]
            n_points = len(z_means)

            if n_points == 1:
                x_positions = [match_pos]
            else:
                x_positions = match_pos + np.linspace(-x_offset, x_offset, n_points)

            ax.scatter(x_positions, z_means, alpha=alpha_per_trace, s=20, color=color,
                       edgecolors='none', zorder=5,
                       label='Observed' if match_pos == 0 else '')

            observed_mean = np.mean(z_means)
            ax.hlines(observed_mean, match_pos - x_offset, match_pos + x_offset,
                      colors=color, linewidth=3, alpha=0.9, zorder=10,
                      label='Observed Mean' if match_pos == 0 else '')

            ax.text(match_pos, ax.get_ylim()[1] * 0.95, f'{n_points}',
                    ha='center', va='top', fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7), zorder=15)

    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, zorder=3)
    ax.set_xlim(-1, num_match_states)
    ax.set_ylim(-3, 3)
    ax.set_xticks(range(0, num_match_states, 5))
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ── Event distribution plots ─────────────────────────────────────────

def _plot_event_distribution(
    results_dict: Dict[str, Dict[str, Any]],
    event_type: str,
    amino_acid: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
    color: str = '#1f77b4',
    show_stats: bool = True
) -> None:
    """Shared implementation for backslip/skip distribution plots."""
    filtered = _filter_results(results_dict, amino_acid)

    all_counts: Dict[int, int] = defaultdict(int)
    for result in filtered.values():
        full_path = result.get('full_path', [])
        if full_path:
            for size, count in _collect_path_events(full_path, event_type).items():
                all_counts[size] += count

    event_label = "Backslip" if event_type == 'backslip' else "Skip"

    if not all_counts:
        logger.info(f"No {event_label.lower()}s found" +
                    (f" for amino acid {amino_acid}" if amino_acid else ""))
        return

    max_size = max(all_counts)
    x_values = list(range(1, max_size + 1))
    y_values = [all_counts.get(i, 0) for i in x_values]

    fig, ax = plt.subplots(figsize=figsize)

    if title is None:
        suffix = amino_acid or "All AAs"
        title = f"{event_label} Distribution - {suffix}"

    ax.set_title(title, fontsize=14, fontweight='bold')
    bars = ax.bar(x_values, y_values, color=color, alpha=0.7, edgecolor='black', linewidth=1)

    for bar, value in zip(bars, y_values):
        if value > 0:
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                    f'{int(value)}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel(f"{event_label} Size (Match State Positions)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Frequency", fontsize=12, fontweight='bold')
    ax.set_xticks(x_values)
    ax.grid(True, axis='y', alpha=0.3)

    if show_stats:
        total = sum(all_counts.values())
        avg_size = sum(s * c for s, c in all_counts.items()) / total

        stats_text = (
            f"Total {event_label}s: {total}\n"
            f"Signals Analyzed: {len(filtered)}\n"
            f"Avg {event_label} Size: {avg_size:.2f}\n"
            f"Max {event_label} Size: {max_size}"
        )
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=10, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_ylim(bottom=0)
    plt.tight_layout()

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
    _plot_event_distribution(results_dict, 'backslip', amino_acid, save_path,
                             figsize, title, color, show_stats)


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
    _plot_event_distribution(results_dict, 'skip', amino_acid, save_path,
                             figsize, title, color, show_stats)


def plot_backslip_by_position(
    results_dict: Dict[str, Dict[str, Any]],
    amino_acid: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6),
    title: Optional[str] = None,
    show_stats: bool = True,
    num_match_states: int = DEFAULT_MATCH_STATES
) -> None:
    """Plot frequency of backslips and skips by match state position."""
    filtered = _filter_results(results_dict, amino_acid)

    backslip_from = defaultdict(int)
    backslip_to = defaultdict(int)
    skip_from = defaultdict(int)
    total_backslips = 0
    total_skips = 0

    for result in filtered.values():
        full_path = result.get('full_path', [])
        if not full_path:
            continue

        prev_match_idx = -1
        for state in full_path:
            if 'Match' not in state:
                continue
            try:
                curr = int(state.split('_')[1])
            except (ValueError, IndexError):
                continue

            if prev_match_idx >= 0:
                if curr < prev_match_idx:
                    backslip_from[prev_match_idx] += 1
                    backslip_to[curr] += 1
                    total_backslips += 1
                elif curr > prev_match_idx + 1:
                    skip_from[prev_match_idx] += 1
                    total_skips += 1

            prev_match_idx = curr

    if total_backslips == 0 and total_skips == 0:
        logger.info("No backslips or skips found" +
                    (f" for amino acid {amino_acid}" if amino_acid else ""))
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    if title is None:
        suffix = amino_acid or "All AAs"
        title = f"Backslip & Skip Frequency by Position - {suffix}"

    fig.suptitle(title, fontsize=14, fontweight='bold')
    x_positions = list(range(num_match_states))

    backslip_values = [backslip_from.get(i, 0) for i in x_positions]
    bars1 = ax1.bar(x_positions, backslip_values, color='#d62728', alpha=0.7,
                    edgecolor='black', linewidth=0.5, label='Backslips FROM')
    for bar, value in zip(bars1, backslip_values):
        if value > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                     f'{int(value)}', ha='center', va='bottom', fontsize=8)

    ax1.set_ylabel("Backslip Frequency", fontsize=11, fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.set_ylim(bottom=0)

    backslip_to_values = [backslip_to.get(i, 0) for i in x_positions]
    ax1_twin = ax1.twinx()
    ax1_twin.plot(x_positions, backslip_to_values, 'o-', color='#ff7f0e',
                  alpha=0.6, linewidth=2, markersize=4, label='Backslips TO')
    ax1_twin.set_ylabel("Backslip TO Frequency", fontsize=11, fontweight='bold', color='#ff7f0e')
    ax1_twin.tick_params(axis='y', labelcolor='#ff7f0e')
    ax1_twin.set_ylim(bottom=0)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

    skip_values = [skip_from.get(i, 0) for i in x_positions]
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
    ax2.set_xticks(range(0, num_match_states, 5))
    ax2.set_xlim(-0.5, num_match_states - 0.5)

    if show_stats and (total_backslips > 0 or total_skips > 0):
        stats_text = f"Total Backslips: {total_backslips}\n"
        if backslip_from:
            max_pos = max(backslip_from, key=backslip_from.get)
            stats_text += f"Most Backslips FROM: Match_{max_pos} ({backslip_from[max_pos]})\n"
        stats_text += f"Total Skips: {total_skips}"
        if skip_from:
            max_skip_pos = max(skip_from, key=skip_from.get)
            stats_text += f"\nMost Skips FROM: Match_{max_skip_pos} ({skip_from[max_skip_pos]})"

        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                 fontsize=9, ha='left', va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()