"""
Visualize full PASTOR across multiple channels
"""

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from config.config import *
from segmentation.filters import apply_bessel_filter
from segmentation.cost_functions import CustomCost
from segmentation.segment_pelt import segment_trace_pelt
from utils.data_loader import load_pastor_data
import ruptures as rpt


def plot_full_pastor(pastor_name, num_channels=8, data_path=DATA_PATH,
                    penalty=PELT_PENALTY, min_size=PELT_MIN_SIZE,
                    scale=PELT_SCALE, cost_function='custom',
                    filter_order=FILTER_ORDER, cutoff=CUTOFF_FREQUENCY,
                    sampling_rate=SAMPLING_RATE, save_path=None):
    """
    Plot segmentation for full PASTOR across multiple channels
    
    Parameters:
    -----------
    pastor_name : str
        PASTOR sequence (e.g., "HDKER")
    num_channels : int
        Number of channels to display
    data_path : str
        Path to data JSON file
    penalty : float
        PELT penalty parameter
    min_size : int
        Minimum segment size
    scale : float
        Scale for custom cost
    cost_function : str
        Cost function to use
    filter_order : int
        Bessel filter order
    cutoff : float
        Filter cutoff frequency
    sampling_rate : float
        Signal sampling rate
    save_path : str
        Path to save figure (optional)
    """
    
    # Load data
    pastor_groups, aa_info, raw_data, channels, run = load_pastor_data(data_path)
    
    if pastor_name not in pastor_groups:
        print(f"PASTOR {pastor_name} not found in data")
        return
    
    # Setup figure
    fig, axes = plt.subplots(num_channels, 5, figsize=FIG_SIZE_FULL)
    
    channel_keys = list(pastor_groups[pastor_name].keys())[:num_channels]
    
    for i, channel_index in enumerate(channel_keys):
        indices = pastor_groups[pastor_name][channel_index][0]
        
        for j in range(len(pastor_name)):
            data = np.asarray(raw_data[indices[j]])
            
            # Filter
            filtered_data = apply_bessel_filter(data, filter_order, cutoff, sampling_rate)
            
            # Segment
            bkps = segment_trace_pelt(filtered_data, penalty, min_size, scale, cost_function)
            
            num_segments = len(bkps) - 1
            ax = axes[i, j]
            
            # Plot signal
            ax.plot(filtered_data, color='black', linewidth=0.8, label='Filtered Signal')
            
            # Add colored segments
            for k in range(num_segments):
                start, end = bkps[k], bkps[k + 1]
                color = COLOR_CYCLE[k % len(COLOR_CYCLE)]
                ax.axvspan(start, end, facecolor=color, alpha=0.3)
            
            ax.set_title(f"{pastor_name} - Ch: {channel_index} - AA: {pastor_name[j]}\nSegs: {num_segments}", 
                        fontsize=10)
            ax.set_xlabel("Sample Index")
            ax.set_ylabel("Current (pA)")
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    plt.suptitle(f"Segmentation for PASTOR {pastor_name} Across {num_channels} Channels", fontsize=16)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot full PASTOR segmentation')
    parser.add_argument('pastor', type=str, help='PASTOR name (e.g., HDKER)')
    parser.add_argument('--channels', type=int, default=8, help='Number of channels to plot')
    parser.add_argument('--penalty', type=float, default=PELT_PENALTY, help='PELT penalty')
    parser.add_argument('--save', type=str, default=None, help='Save path for figure')
      
    args = parser.parse_args()
    
    plot_full_pastor(args.pastor, num_channels=args.channels, 
                    penalty=args.penalty, save_path=args.save)