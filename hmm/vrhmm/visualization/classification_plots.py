"""Classification visualization utilities."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

logger = logging.getLogger(__name__)


def generate_classification_report(
        results_df: pd.DataFrame,
        classification_mode: str,
        output_dir: str
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        plot_confusion_matrix(
            results_df,
            classification_mode,
            output_path / f'confusion_matrix_{classification_mode}.pdf'
        )

        plot_category_performance(
            results_df,
            classification_mode,
            output_path / f'category_performance_{classification_mode}.pdf'
        )

        save_text_report(
            results_df,
            classification_mode,
            output_path / f'classification_report_{classification_mode}.txt'
        )

    except Exception as e:
        logger.error(f"Error generating classification plots: {e}")


def plot_confusion_matrix(
        df: pd.DataFrame,
        mode: str,
        save_path: Path
) -> None:
    categories = sorted(df['true_category'].unique())

    cm = confusion_matrix(
        df['true_category'],
        df['predicted_category'],
        labels=categories
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        square=True,
        xticklabels=categories,
        yticklabels=categories,
        ax=ax
    )

    accuracy = (df['true_category'] == df['predicted_category']).mean() * 100
    balanced_acc = balanced_accuracy_score(df['true_category'], df['predicted_category']) * 100
    
    ax.set_title(f'{mode.upper()} Classification\nAccuracy: {accuracy:.1f}% | Balanced Accuracy: {balanced_acc:.1f}%')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_category_performance(
        df: pd.DataFrame,
        mode: str,
        save_path: Path
) -> None:
    categories = df['true_category'].unique()

    metrics = []
    for category in categories:
        mask = df['true_category'] == category
        if mask.sum() > 0:
            accuracy = (
                df[mask]['true_category'] == df[mask]['predicted_category']
            ).mean() * 100
            metrics.append({'Category': category, 'Accuracy': accuracy, 'Count': mask.sum()})

    df_metrics = pd.DataFrame(metrics)
    df_metrics = df_metrics.sort_values('Accuracy')

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(df_metrics['Category'], df_metrics['Accuracy'])

    for i, (acc, cnt) in enumerate(zip(df_metrics['Accuracy'], df_metrics['Count'])):
        ax.text(acc + 1, i, f'n={cnt}', va='center')

    overall_acc = (df['true_category'] == df['predicted_category']).mean() * 100
    balanced_acc = balanced_accuracy_score(df['true_category'], df['predicted_category']) * 100
    ax.axvline(overall_acc, color='red', linestyle='--', label=f'Overall: {overall_acc:.1f}%')
    ax.axvline(balanced_acc, color='blue', linestyle=':', label=f'Balanced: {balanced_acc:.1f}%')

    ax.set_xlabel('Accuracy (%)')
    ax.set_ylabel('Category')
    ax.set_xlim(0, 105)
    ax.set_title(f'{mode.upper()} Per-Category Performance')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_text_report(
        df: pd.DataFrame,
        mode: str,
        save_path: Path
) -> None:
    report = classification_report(
        df['true_category'],
        df['predicted_category'],
        digits=3
    )

    accuracy = (df['true_category'] == df['predicted_category']).mean() * 100
    balanced_acc = balanced_accuracy_score(df['true_category'], df['predicted_category']) * 100

    with open(save_path, 'w') as f:
        f.write(f"Classification Report - {mode.upper()}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Accuracy:          {accuracy:.1f}%\n")
        f.write(f"Balanced Accuracy: {balanced_acc:.1f}%\n\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)


def plot_triangular_correlation_matrix(
        correlation_df: pd.DataFrame,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10),
        cmap: str = 'RdYlGn',
        title: str = 'Pairwise Classification Matrix',
        vmin: float = 0.5,
        vmax: float = 1.0,
        annotate: bool = True,
        annotation_fontsize: int = 8
) -> None:
    """
    Plot a triangular correlation-style matrix using seaborn.
    
    This creates a lower-triangular heatmap suitable for showing
    pairwise relationships (accuracy, similarity, etc.) between
    amino acids or categories.
    """
    mask = np.triu(np.ones_like(correlation_df, dtype=bool), k=0)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        correlation_df,
        mask=mask,
        annot=annotate,
        fmt='.2f',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=True,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'Value', 'shrink': 0.8},
        ax=ax,
        annot_kws={'size': annotation_fontsize}
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved triangular matrix to: {save_path}")
    
    plt.close()


def plot_pairwise_heatmap_with_dendrograms(
        matrix_df: pd.DataFrame,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 12),
        cmap: str = 'RdYlGn',
        title: str = 'Clustered Pairwise Classification Matrix'
) -> None:
    """
    Plot a clustered heatmap with dendrograms showing hierarchical
    relationships between amino acids based on classification similarity.
    """
    matrix_filled = matrix_df.fillna(0.5)
    
    g = sns.clustermap(
        matrix_filled,
        cmap=cmap,
        vmin=0.5,
        vmax=1.0,
        annot=True,
        fmt='.2f',
        square=True,
        linewidths=0.5,
        figsize=figsize,
        dendrogram_ratio=(0.15, 0.15),
        cbar_pos=(0.02, 0.8, 0.03, 0.15),
        annot_kws={'size': 7}
    )
    
    g.fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    if save_path:
        g.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved clustered heatmap to: {save_path}")
    
    plt.close()


def create_amino_acid_similarity_network(
        accuracy_matrix: pd.DataFrame,
        threshold: float = 0.7,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 12)
) -> Dict[str, Any]:
    """
    Create a network visualization where amino acids are nodes
    and edges represent high confusion (low pairwise accuracy).
    
    Returns network statistics and optionally saves the plot.
    """
    try:
        import networkx as nx
    except ImportError:
        logger.warning("networkx not available for network visualization")
        return {}
    
    G = nx.Graph()
    
    amino_acids = list(accuracy_matrix.index)
    G.add_nodes_from(amino_acids)
    
    edges = []
    for i, aa1 in enumerate(amino_acids):
        for j, aa2 in enumerate(amino_acids):
            if i < j:
                acc = accuracy_matrix.loc[aa1, aa2]
                if not np.isnan(acc) and acc < threshold:
                    weight = 1 - acc
                    edges.append((aa1, aa2, {'weight': weight, 'accuracy': acc}))
    
    G.add_edges_from(edges)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    degrees = dict(G.degree())
    node_sizes = [300 + 100 * degrees[node] for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue',
                           edgecolors='black', linewidths=1.5, ax=ax)
    
    if edges:
        edge_weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]
        edge_colors = [G[u][v]['accuracy'] for u, v in G.edges()]
        
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.6,
                               edge_color=edge_colors, edge_cmap=plt.cm.Reds_r,
                               edge_vmin=0.5, edge_vmax=threshold, ax=ax)
    
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
    
    ax.set_title(f'Amino Acid Confusion Network (threshold < {threshold})',
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved network visualization to: {save_path}")
    
    plt.close()
    
    components = list(nx.connected_components(G))
    
    stats = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'num_components': len(components),
        'components': [list(c) for c in components],
        'most_confused': sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    }
    
    return stats