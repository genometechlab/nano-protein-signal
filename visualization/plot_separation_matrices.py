"""
Visualize pairwise separation matrices
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.append('..')
from config.config import IDX_TO_AA


def plot_separation_matrix(matrix, AA_LIST, title, output_path=None,
                           mask_diagonal=True, cmap='RdYlGn', vmin=0.5, vmax=2.0):
    """
    Plot separation matrix
    
    Parameters:
    -----------
    matrix : np.ndarray
        Separation ratio matrix
    AA_LIST : list
        Amino acid labels
    title : str
        Plot title
    output_path : str, optional
        Save path
    mask_diagonal : bool
        Mask upper triangle and diagonal
    cmap : str
        Colormap
    vmin : float
        Minimum value for colorscale
    vmax : float
        Maximum value for colorscale
    """
    n = len(AA_LIST)
    
    plot_matrix = matrix.copy()
    
    if mask_diagonal:
        # Mask upper triangle including diagonal
        mask = np.triu(np.ones_like(matrix, dtype=bool), k=0)
        plot_matrix = np.ma.masked_array(matrix, mask=mask)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(plot_matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(AA_LIST, fontsize=9)
    ax.set_yticklabels(AA_LIST, fontsize=9)
    ax.set_xlabel("Trace Class", fontsize=11)
    ax.set_ylabel("Centroid Class", fontsize=11)
    ax.set_title(title, fontsize=12)
    
    plt.colorbar(im, ax=ax, shrink=0.8, label="Separation Ratio (inter/intra)")
    
    # Add text annotations for lower triangle
    for i in range(n):
        for j in range(i):
            val = matrix[i, j]
            color = 'white' if val < 1.2 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                   fontsize=6, color=color)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.show()


def plot_separation_histogram(separation_matrix, AA_LIST, metric_name,
                              output_path=None):
    """
    Plot histogram of separation ratios
    
    Parameters:
    -----------
    separation_matrix : np.ndarray
        Separation ratio matrix
    AA_LIST : list
        Amino acid labels
    metric_name : str
        Metric name for title
    output_path : str, optional
        Save path
    """
    n = len(AA_LIST)
    mask = np.tril(np.ones((n, n), dtype=bool), k=-1)
    vals = separation_matrix[mask]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.hist(vals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2,
              label='No separation (ratio=1)')
    ax.axvline(x=np.median(vals), color='blue', linestyle='-', linewidth=2,
              label=f'Median: {np.median(vals):.2f}')
    
    ax.set_xlabel("Separation Ratio (inter/intra)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"{metric_name} Separation Distribution\n(% > 1: {100*np.mean(vals > 1):.0f}%)",
                fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.show()


def plot_all_separation_results(results_dir, AA_LIST=None):
    """
    Load and plot all separation results
    
    Parameters:
    -----------
    results_dir : str
        Directory containing separation analysis results
    AA_LIST : list, optional
        Amino acid list (will infer from matrix size if None)
    """
    results_dir = Path(results_dir)
    
    # Load matrices
    dev_sep = np.load(results_dir / "dev_separation.npy")
    cost_sep = np.load(results_dir / "cost_separation.npy")
    combined_sep = np.load(results_dir / "combined_separation.npy")
    
    if AA_LIST is None:
        n = dev_sep.shape[0]
        AA_LIST = [IDX_TO_AA[i] for i in range(n)]
    
    # Plot matrices
    plot_separation_matrix(
        dev_sep, AA_LIST,
        "Path Deviation Separation (inter/intra, >1 = separable)",
        output_path=results_dir / "deviation_separation_matrix.png"
    )
    
    plot_separation_matrix(
        cost_sep, AA_LIST,
        "DTW Cost Separation (inter/intra, >1 = separable)",
        output_path=results_dir / "cost_separation_matrix.png"
    )
    
    plot_separation_matrix(
        combined_sep, AA_LIST,
        "Combined Separation (inter/intra, >1 = separable)",
        output_path=results_dir / "combined_separation_matrix.png"
    )
    
    # Plot histograms
    plot_separation_histogram(
        dev_sep, AA_LIST, "Path Deviation",
        output_path=results_dir / "deviation_histogram.png"
    )
    
    plot_separation_histogram(
        cost_sep, AA_LIST, "DTW Cost",
        output_path=results_dir / "cost_histogram.png"
    )
    
    plot_separation_histogram(
        combined_sep, AA_LIST, "Combined",
        output_path=results_dir / "combined_histogram.png"
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot separation analysis results')
    parser.add_argument('results_dir', type=str, help='Results directory')
    parser.add_argument('--aas', type=str, nargs='+', default=None,
                       help='Amino acid labels')
    
    args = parser.parse_args()
    
    plot_all_separation_results(args.results_dir, args.aas)