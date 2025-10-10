"""
Visualize DTW alignment between traces or trace vs barycenter
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import pickle
import json
from pathlib import Path

import sys
sys.path.append('..')

from utils.dtw_utils import (
    dtw_matrix, zscore_both, compute_cost_matrix, compute_alignment_metrics
)
def concat_plot_with_links(segs1, segs2, path, ax_top, ax_bot, label1="Trace 1", label2="Trace 2"):
    """
    Plot two traces with alignment links
    
    Parameters:
    -----------
    segs1 : list of arrays
        First trace segments
    segs2 : list of arrays
        Second trace segments
    path : list of tuples
        Alignment path
    ax_top : matplotlib axis
        Top axis for first trace
    ax_bot : matplotlib axis
        Bottom axis for second trace
    label1 : str
        Label for first trace
    label2 : str
        Label for second trace
    """
    # Plot first trace
    concat1 = np.concatenate(segs1)
    ax_top.plot(concat1, lw=1, color='red')
    
    # Plot second trace
    pos = 0
    for seg in segs2:
        ax_bot.plot(range(pos, pos + len(seg)), seg, lw=0.8, color='blue')
        pos += len(seg)
    
    # Add connection lines
    fig = ax_top.get_figure()
    for i, j in path:
        xi = sum(len(s) for s in segs1[:i]) + len(segs1[i]) / 2
        yi = segs1[i][-1]
        xj = sum(len(s) for s in segs2[:j]) + len(segs2[j]) / 2
        yj = segs2[j][-1]
        fig.add_artist(ConnectionPatch(
            (xi, yi), (xj, yj),
            "data", "data",
            axesA=ax_top, axesB=ax_bot,
            ls='--', lw=0.6, color='k', alpha=0.5
        ))
    
    # Set x-axis
    maxlen = max(len(concat1), sum(len(s) for s in segs2))
    ticks = np.arange(0, maxlen + 1, 250)
    for ax in (ax_top, ax_bot):
        ax.set_xlim(-10, maxlen + 10)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
    
    ax_top.set_ylabel(label1)
    ax_bot.set_ylabel(label2)
    ax_bot.set_xlabel("Sample Index")


def plot_dtw_alignment(segs1, segs2, aa_label, metadata1="Reference", metadata2="Test",
                      trace_id=None, save_path=None):
    """
    Create DTW alignment visualization
    
    Parameters:
    -----------
    segs1 : list of arrays
        First trace segments
    segs2 : list of arrays
        Second trace segments
    aa_label : str
        Amino acid label
    metadata1 : str
        Metadata for first trace (e.g., "Barycenter" or "Pastor_Channel")
    metadata2 : str
        Metadata for second trace
    trace_id : int, optional
        Trace ID for labeling
    save_path : str, optional
        Path to save figure
    """
    # Normalize
    segs1_z, segs2_z = zscore_both(segs1, segs2)
    
    # Compute cost matrix
    cost_mat = compute_cost_matrix(segs1_z, segs2_z)
    
    # Compute alignment
    dp, path = dtw_matrix(segs1_z, segs2_z, cost_mat)
    
    # Compute metrics
    metrics = compute_alignment_metrics(dp, path)
    
    # Create figure
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[5, 4])
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_mat = fig.add_subplot(gs[:, 1])
    
    # Plot traces with links
    concat_plot_with_links(segs1_z, segs2_z, path, ax_top, ax_bot, metadata1, metadata2)
    
    # Plot cost matrix with path
    im = ax_mat.imshow(cost_mat, cmap="viridis", aspect="auto")
    ax_mat.plot(*zip(*[(j, i) for i, j in path]), 'r.-', ms=3, lw=1)
    ax_mat.set_xlabel(f"{metadata2} Segment #")
    ax_mat.set_ylabel(f"{metadata1} Segment #")
    fig.colorbar(im, ax=ax_mat, label="Segment Cost")
    
    # Title
    title_parts = [f"AA: {aa_label}", metadata1, "vs", metadata2]
    if trace_id is not None:
        title_parts.append(f"Trace #{trace_id}")
    
    title = " | ".join(title_parts)
    subtitle = (f"Total DTW={metrics['total_cost']:.2f} | "
               f"Mean DTW={metrics['mean_cost']:.2f} | "
               f"Deviation={metrics['deviation']:.2f}")
    
    fig.suptitle(f"{title}\n{subtitle}", fontsize=13)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def load_traces_from_pickle(pickle_file, aa_label, pastor=None, channel=None, 
                            run=None, segment_key='segments'):
    """
    Load traces from pickle file with optional filtering
    
    Parameters:
    -----------
    pickle_file : str
        Path to pickle file
    aa_label : str
        Amino acid to filter
    pastor : str, optional
        Filter by pastor
    channel : int, optional
        Filter by channel
    run : str, optional
        Filter by run
    segment_key : str
        Key for segments in pickle data ('segments' or 'cleaned_segments')
    
    Returns:
    --------
    trace_data : list of tuples
        List of (segments, metadata_dict)
    """
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)
    
    trace_data = []
    for e in data:
        if e["variable_region"] != aa_label:
            continue
        
        # Get metadata
        metadata_str = e.get("metadata", "")
        e_pastor = metadata_str.split('_')[0] if '_' in metadata_str else 'Unknown'
        e_channel = e.get("channel", None)
        e_run = e.get("run", None)
        
        # Apply filters
        if pastor is not None and e_pastor != pastor:
            continue
        if channel is not None and e_channel != channel:
            continue
        if run is not None and e_run != run:
            continue
        
        segments = e.get(segment_key, e.get('cleaned_segments', []))
        segments = [np.array(s) for s in segments]
        
        metadata_dict = {
            'pastor': e_pastor,
            'channel': e_channel,
            'run': e_run,
            'metadata': metadata_str
        }
        
        trace_data.append((segments, metadata_dict))
    
    return trace_data


def visualize_trace_vs_barycenter(pickle_file, centroid_json, aa_label,
                                  pastor=None, channel=None, run=None,
                                  save_pdf=False, output_dir=None):
    """
    Visualize traces against barycenter
    
    Parameters:
    -----------
    pickle_file : str
        Path to segmented pickle file
    centroid_json : str
        Path to centroid JSON file
    aa_label : str
        Amino acid label
    pastor : str, optional
        Filter by pastor
    channel : int, optional
        Filter by channel
    run : str, optional
        Filter by run
    save_pdf : bool
        Whether to save PDFs
    output_dir : str, optional
        Output directory for PDFs
    """
    # Load centroid
    with open(centroid_json) as f:
        all_centroids = json.load(f)
    
    if aa_label not in all_centroids:
        raise ValueError(f"No centroid for amino acid '{aa_label}'")
    
    centroid_segs = [np.array(s) for s in all_centroids[aa_label]]
    
    # Load traces
    trace_data = load_traces_from_pickle(pickle_file, aa_label, pastor, channel, run)
    
    if not trace_data:
        print(f"No traces found for AA '{aa_label}' with specified filters")
        return
    
    print(f"Found {len(trace_data)} traces matching criteria")
    
    # Create output directory
    if save_pdf and output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Visualize each trace
    for test_id, (test_segs, metadata) in enumerate(trace_data):
        metadata_str = f"Pastor {metadata['pastor']} | Channel {metadata['channel']}"
        
        print(f"\nTrace #{test_id}: {metadata_str}")
        
        save_path = None
        if save_pdf:
            filename = f"dtw_AA{aa_label}_Pastor{metadata['pastor']}_Ch{metadata['channel']}_trace{test_id}.pdf"
            if output_dir:
                save_path = str(Path(output_dir) / filename)
            else:
                save_path = filename
        
        plot_dtw_alignment(
            centroid_segs, test_segs,
            aa_label=aa_label,
            metadata1="Barycenter",
            metadata2=metadata_str,
            trace_id=test_id,
            save_path=save_path
        )


def visualize_trace_vs_trace(pickle_file, aa_label, trace_idx1, trace_idx2,
                             pastor=None, channel=None, run=None,
                             save_path=None):
    """
    Visualize alignment between two traces
    
    Parameters:
    -----------
    pickle_file : str
        Path to segmented pickle file
    aa_label : str
        Amino acid label
    trace_idx1 : int
        Index of first trace
    trace_idx2 : int
        Index of second trace
    pastor : str, optional
        Filter by pastor
    channel : int, optional
        Filter by channel
    run : str, optional
        Filter by run
    save_path : str, optional
        Path to save figure
    """
    # Load traces
    trace_data = load_traces_from_pickle(pickle_file, aa_label, pastor, channel, run)
    
    if not trace_data:
        print(f"No traces found for AA '{aa_label}'")
        return
    
    if trace_idx1 >= len(trace_data) or trace_idx2 >= len(trace_data):
        print(f"Invalid trace indices. Available: 0-{len(trace_data)-1}")
        return
    
    segs1, meta1 = trace_data[trace_idx1]
    segs2, meta2 = trace_data[trace_idx2]
    
    meta1_str = f"Trace {trace_idx1} | Pastor {meta1['pastor']} | Ch {meta1['channel']}"
    meta2_str = f"Trace {trace_idx2} | Pastor {meta2['pastor']} | Ch {meta2['channel']}"
    
    print(f"\nComparing:\n  {meta1_str}\n  {meta2_str}")
    
    plot_dtw_alignment(
        segs1, segs2,
        aa_label=aa_label,
        metadata1=meta1_str,
        metadata2=meta2_str,
        save_path=save_path
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize DTW alignment')
    parser.add_argument('pickle', type=str, help='Path to segmented pickle file')
    parser.add_argument('aa', type=str, help='Amino acid (e.g., D)')
    parser.add_argument('--mode', type=str, choices=['barycenter', 'trace'], default='barycenter',
                       help='Comparison mode: barycenter or trace-to-trace')
    
    # For barycenter mode
    parser.add_argument('--centroid', type=str, default=None,
                       help='Path to centroid JSON (required for barycenter mode)')
    
    # For trace-to-trace mode
    parser.add_argument('--trace1', type=int, default=None,
                       help='First trace index (for trace mode)')
    parser.add_argument('--trace2', type=int, default=None,
                       help='Second trace index (for trace mode)')
    
    # Filters
    parser.add_argument('--pastor', type=str, default=None, help='Filter by pastor')
    parser.add_argument('--channel', type=int, default=None, help='Filter by channel')
    parser.add_argument('--run', type=str, default=None, help='Filter by run')
    
    # Output
    parser.add_argument('--save-pdf', action='store_true', help='Save PDFs')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    parser.add_argument('--save', type=str, default=None, help='Save path for single figure')
    
    args = parser.parse_args()
    
    if args.mode == 'barycenter':
        if args.centroid is None:
            parser.error("--centroid is required for barycenter mode")
        
        visualize_trace_vs_barycenter(
            args.pickle, args.centroid, args.aa,
            pastor=args.pastor, channel=args.channel, run=args.run,
            save_pdf=args.save_pdf, output_dir=args.output_dir
        )
    
    elif args.mode == 'trace':
        if args.trace1 is None or args.trace2 is None:
            parser.error("--trace1 and --trace2 are required for trace mode")
        
        visualize_trace_vs_trace(
            args.pickle, args.aa, args.trace1, args.trace2,
            pastor=args.pastor, channel=args.channel, run=args.run,
            save_path=args.save
        )