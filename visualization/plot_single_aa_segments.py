"""
Visualize single amino acid from specific PASTOR and channel
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


def plot_single_aa(pastor_name, target_aa, target_channel, data_path=DATA_PATH,
                  penalty=PELT_PENALTY, min_size=PELT_MIN_SIZE,
                  scale=PELT_SCALE, cost_function='custom',
                  filter_order=FILTER_ORDER, cutoff=CUTOFF_FREQUENCY,
                  sampling_rate=SAMPLING_RATE, save_path=None):
    """
    Plot segmentation for single amino acid
    
    Parameters:
    -----------
    pastor_name : str
        PASTOR sequence (e.g., "HDKER")
    target_aa : str
        Target amino acid (e.g., "D")
    target_channel : int
        Channel number
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
    
    if target_aa not in pastor_name:
        print(f"Amino acid {target_aa} not in PASTOR {pastor_name}")
        return
    
    aa_position = pastor_name.index(target_aa)
    
    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE_SINGLE)
    
    if target_channel in pastor_groups[pastor_name]:
        indices = pastor_groups[pastor_name][target_channel][0]
        data = np.asarray(raw_data[indices[aa_position]])
        
        # Filter
        filtered_data = apply_bessel_filter(data, filter_order, cutoff, sampling_rate)
        
        # Segment
        bkps = segment_trace_pelt(filtered_data, penalty, min_size, scale, cost_function)
        
        num_segments = len(bkps) - 1
        
        # Plot signal
        ax.plot(filtered_data, color='black', linewidth=1, label='Filtered Signal')
        
        # Add colored segments
        for k in range(num_segments):
            start, end = bkps[k], bkps[k + 1]
            color = COLOR_CYCLE[k % len(COLOR_CYCLE)]
            ax.axvspan(start, end, facecolor=color, alpha=0.3)
        
        ax.set_title(f"PASTOR: {pastor_name} | Channel: {target_channel} | AA: {target_aa} | Segments: {num_segments}", 
                    fontsize=14)
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Current (pA)")
    else:
        ax.text(0.5, 0.5, f'No data for Channel {target_channel}', 
               transform=ax.transAxes, ha='center', va='center')
        ax.set_title(f"PASTOR: {pastor_name} | Channel: {target_channel} | AA: {target_aa} - No Data", 
                    fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot single amino acid segmentation')
    parser.add_argument('pastor', type=str, help='PASTOR name (e.g., HDKER)')
    parser.add_argument('aa', type=str, help='Amino acid (e.g., D)')
    parser.add_argument('channel', type=int, help='Channel number')
    parser.add_argument('--penalty', type=float, default=PELT_PENALTY, help='PELT penalty')
    parser.add_argument('--save', type=str, default=None, help='Save path for figure')
    
    args = parser.parse_args()
    
    plot_single_aa(args.pastor, args.aa, args.channel, 
                  penalty=args.penalty, save_path=args.save)