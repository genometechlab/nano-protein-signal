"""
Visualize DBA centroids
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.append('..')
from config.config import FIXED_YLIM
from dtw.barycenter import load_centroids


def plot_dba_centroids(centroid_path, output_path=None, ylim=FIXED_YLIM):
    """
    Plot DBA centroids for all amino acids
    
    Parameters:
    -----------
    centroid_path : str
        Path to centroid JSON file
    output_path : str
        Path to save figure
    ylim : tuple
        Y-axis limits
    """
    
    centroids = load_centroids(centroid_path)
    AA_LIST = sorted(centroids.keys())
    
    fig, axes = plt.subplots(len(AA_LIST), 1, figsize=(12, len(AA_LIST) * 1.2), sharex=True)
    if len(AA_LIST) == 1:
        axes = [axes]
    
    for i, aa in enumerate(AA_LIST):
        trace = np.concatenate(centroids[aa])
        ax = axes[i]
        ax.plot(trace, lw=0.8, color='steelblue')
        ax.set_ylim(*ylim)
        ax.set_ylabel(aa, rotation=0, labelpad=20, fontsize=11, weight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yticks([])
        
        if i < len(AA_LIST) - 1:
            ax.set_xticks([])
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
    
    axes[-1].set_xlabel("Sample Index", fontsize=12)
    plt.suptitle("DBA Centroids for All Amino Acids", fontsize=14, y=0.995)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot DBA centroids')
    parser.add_argument('centroids', type=str, help='Path to centroid JSON file')
    parser.add_argument('--save', type=str, default=None, help='Save path for figure')
    parser.add_argument('--ylim', type=float, nargs=2, default=FIXED_YLIM,
                       help='Y-axis limits (min max)')
    
    args = parser.parse_args()
    
    plot_dba_centroids(args.centroids, args.save, tuple(args.ylim))