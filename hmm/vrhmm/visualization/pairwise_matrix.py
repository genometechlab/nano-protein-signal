"""Pairwise classification matrix visualization."""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

logger = logging.getLogger(__name__)

AMINO_ACID_ORDER = list('ACDEFGHIKLMNPQRSTVWY')


# ── Helpers ───────────────────────────────────────────────────────────

def _discover_amino_acids(
    results_dict: Dict[str, Dict[str, Any]],
    amino_acids: Optional[List[str]] = None
) -> List[str]:
    """Return a sorted list of amino acids present in results."""
    if amino_acids is not None:
        return amino_acids
    return sorted(set(
        r.get('amino_acid', '') for r in results_dict.values()
        if r.get('amino_acid', '') in AMINO_ACID_ORDER
    ))


def _group_by_amino_acid(
    results_dict: Dict[str, Dict[str, Any]],
    amino_acids: List[str]
) -> Dict[str, List[Dict[str, Any]]]:
    """Group results by amino acid label."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for result in results_dict.values():
        aa = result.get('amino_acid', '')
        if aa in amino_acids:
            grouped[aa].append(result)
    return grouped


def _report_path(save_path: str) -> str:
    """Derive a text report path from a figure path."""
    return str(Path(save_path).with_suffix('.txt').parent / (Path(save_path).stem + '_report.txt'))


# ── Computation ───────────────────────────────────────────────────────

def compute_pairwise_classification_accuracy(
    results_dict: Dict[str, Dict[str, Any]],
    classifier: Any,
    amino_acids: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], Dict[str, Any]]]:
    """Compute pairwise classification accuracy between all amino acid pairs."""
    amino_acids = _discover_amino_acids(results_dict, amino_acids)
    aa_signals = _group_by_amino_acid(results_dict, amino_acids)

    n_aa = len(amino_acids)
    accuracy_matrix = np.full((n_aa, n_aa), np.nan)
    pair_details = {}

    for i, aa1 in enumerate(amino_acids):
        accuracy_matrix[i, i] = 1.0
        for j, aa2 in enumerate(amino_acids):
            if i >= j:
                continue

            signals_aa1 = aa_signals.get(aa1, [])
            signals_aa2 = aa_signals.get(aa2, [])

            if not signals_aa1 or not signals_aa2:
                continue

            correct_aa1 = total_aa1 = 0
            correct_aa2 = total_aa2 = 0

            for result in signals_aa1:
                scores = result.get('all_scores', {})
                if aa1 in scores and aa2 in scores:
                    if scores[aa1] > scores[aa2]:
                        correct_aa1 += 1
                    total_aa1 += 1

            for result in signals_aa2:
                scores = result.get('all_scores', {})
                if aa1 in scores and aa2 in scores:
                    if scores[aa2] > scores[aa1]:
                        correct_aa2 += 1
                    total_aa2 += 1

            total = total_aa1 + total_aa2
            correct = correct_aa1 + correct_aa2

            if total > 0:
                acc = correct / total
                accuracy_matrix[i, j] = acc
                accuracy_matrix[j, i] = acc

                pair_details[(aa1, aa2)] = {
                    'accuracy': acc,
                    'total': total,
                    'correct': correct,
                    f'{aa1}_correct': correct_aa1,
                    f'{aa1}_total': total_aa1,
                    f'{aa2}_correct': correct_aa2,
                    f'{aa2}_total': total_aa2,
                    f'{aa1}_accuracy': correct_aa1 / total_aa1 if total_aa1 > 0 else np.nan,
                    f'{aa2}_accuracy': correct_aa2 / total_aa2 if total_aa2 > 0 else np.nan
                }

    return pd.DataFrame(accuracy_matrix, index=amino_acids, columns=amino_acids), pair_details


def compute_pairwise_log_likelihood_ratio(
    results_dict: Dict[str, Dict[str, Any]],
    amino_acids: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], Dict[str, Any]]]:
    """Compute average log-likelihood ratio for pairwise discrimination."""
    amino_acids = _discover_amino_acids(results_dict, amino_acids)
    aa_signals = _group_by_amino_acid(results_dict, amino_acids)

    n_aa = len(amino_acids)
    llr_matrix = np.full((n_aa, n_aa), np.nan)
    pair_details = {}

    for i, aa1 in enumerate(amino_acids):
        llr_matrix[i, i] = 0.0
        for j, aa2 in enumerate(amino_acids):
            if i == j:
                continue

            signals = aa_signals.get(aa1, [])
            if not signals:
                continue

            llr_values = []
            for result in signals:
                scores = result.get('all_scores', {})
                if aa1 in scores and aa2 in scores:
                    llr_values.append(scores[aa1] - scores[aa2])

            if llr_values:
                mean_llr = np.mean(llr_values)
                llr_matrix[i, j] = mean_llr

                if i < j:
                    pair_details[(aa1, aa2)] = {
                        f'{aa1}_vs_{aa2}_mean_llr': mean_llr,
                        f'{aa1}_vs_{aa2}_std_llr': np.std(llr_values),
                        f'{aa1}_vs_{aa2}_n': len(llr_values)
                    }

    return pd.DataFrame(llr_matrix, index=amino_acids, columns=amino_acids), pair_details


# ── Reporting ─────────────────────────────────────────────────────────

def save_pairwise_report(
    accuracy_matrix: pd.DataFrame,
    pair_details: Dict[Tuple[str, str], Dict[str, Any]],
    save_path: str,
    title: str = "Pairwise Classification Report"
) -> None:
    """Save a detailed text report of pairwise classification results."""
    pairs_list = []
    for (aa1, aa2), details in pair_details.items():
        pairs_list.append({'pair': f'{aa1}-{aa2}', 'aa1': aa1, 'aa2': aa2, **details})

    df_pairs = pd.DataFrame(pairs_list)
    if 'accuracy' in df_pairs.columns:
        df_pairs = df_pairs.sort_values('accuracy', ascending=True)

    with open(save_path, 'w') as f:
        f.write(f"{title}\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        if 'accuracy' in df_pairs.columns:
            acc_values = df_pairs['accuracy'].dropna()
            f.write("SUMMARY STATISTICS\n")
            f.write(f"  Total pairs analyzed: {len(acc_values)}\n")
            f.write(f"  Mean pairwise accuracy: {acc_values.mean():.4f}\n")
            f.write(f"  Median pairwise accuracy: {acc_values.median():.4f}\n")
            f.write(f"  Std pairwise accuracy: {acc_values.std():.4f}\n")
            f.write(f"  Min pairwise accuracy: {acc_values.min():.4f}\n")
            f.write(f"  Max pairwise accuracy: {acc_values.max():.4f}\n")
            f.write(f"  Pairs below 60%: {(acc_values < 0.6).sum()}\n")
            f.write(f"  Pairs below 70%: {(acc_values < 0.7).sum()}\n")
            f.write(f"  Pairs above 90%: {(acc_values >= 0.9).sum()}\n")
            f.write(f"  Pairs above 95%: {(acc_values >= 0.95).sum()}\n\n")

        worst_n = min(20, len(df_pairs))
        f.write("MOST CONFUSED PAIRS (lowest accuracy)\n")
        for _, row in df_pairs.head(worst_n).iterrows():
            if 'accuracy' in row:
                n_str = f" (n={row['total']})" if 'total' in row else ""
                f.write(f"  {row['pair']}: {row['accuracy']:.4f}{n_str}\n")
        f.write("\n")

        best_n = min(20, len(df_pairs))
        f.write("BEST SEPARATED PAIRS (highest accuracy)\n")
        for _, row in df_pairs.tail(best_n).iloc[::-1].iterrows():
            if 'accuracy' in row:
                n_str = f" (n={row['total']})" if 'total' in row else ""
                f.write(f"  {row['pair']}: {row['accuracy']:.4f}{n_str}\n")
        f.write("\n")

        f.write("PER-AMINO-ACID DISCRIMINATION\n")
        aa_stats: Dict[str, List[float]] = defaultdict(list)
        for (aa1, aa2), details in pair_details.items():
            if 'accuracy' in details:
                aa_stats[aa1].append(details['accuracy'])
                aa_stats[aa2].append(details['accuracy'])

        aa_means = sorted(
            [(aa, np.mean(accs), len(accs)) for aa, accs in aa_stats.items()],
            key=lambda x: x[1]
        )
        for aa, mean_acc, n_pairs in aa_means:
            f.write(f"  {aa}: mean accuracy {mean_acc:.4f} across {n_pairs} pairs\n")
        f.write("\n")

        f.write("DETAILED PAIR STATISTICS\n")
        for _, row in df_pairs.iterrows():
            f.write(f"\n  {row['pair']}:\n")
            for col in df_pairs.columns:
                if col in ('pair', 'aa1', 'aa2'):
                    continue
                val = row[col]
                if pd.notna(val):
                    fmt = f"{val:.4f}" if isinstance(val, float) else str(val)
                    f.write(f"    {col}: {fmt}\n")

        f.write("\n\nFULL ACCURACY MATRIX\n")
        f.write(accuracy_matrix.to_string())
        f.write("\n")

    logger.info(f"Saved pairwise report to: {save_path}")


# ── Plotting ──────────────────────────────────────────────────────────

def plot_pairwise_classification_matrix(
    results_dict: Dict[str, Dict[str, Any]],
    classifier: Any = None,
    metric: str = 'accuracy',
    amino_acids: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = 'RdYlGn',
    title: Optional[str] = None,
    annotate: bool = True,
    mask_diagonal: bool = True,
    triangular: bool = True,
    save_text_report: bool = True
) -> pd.DataFrame:
    """Create a pairwise classification matrix heatmap. Returns the matrix DataFrame."""
    amino_acids = _discover_amino_acids(results_dict, amino_acids)

    if metric == 'accuracy':
        matrix_df, pair_details = compute_pairwise_classification_accuracy(
            results_dict, classifier, amino_acids
        )
        vmin, vmax = 0.5, 1.0
        fmt = '.2f'
        default_title = 'Pairwise Classification Accuracy'
        cbar_label = 'Accuracy'
    elif metric == 'llr':
        matrix_df, pair_details = compute_pairwise_log_likelihood_ratio(results_dict, amino_acids)
        abs_max = np.nanmax(np.abs(matrix_df.values))
        vmin, vmax = -abs_max, abs_max
        fmt = '.1f'
        default_title = 'Pairwise Log-Likelihood Ratio'
        cbar_label = 'Mean LLR'
        cmap = 'coolwarm'
    else:
        raise ValueError(f"Unknown metric: {metric}")

    mask = None
    if triangular:
        mask = np.triu(np.ones_like(matrix_df, dtype=bool), k=1)
    if mask_diagonal:
        if mask is None:
            mask = np.eye(len(amino_acids), dtype=bool)
        else:
            np.fill_diagonal(mask, True)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        matrix_df,
        mask=mask,
        annot=annotate,
        fmt=fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=True,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': cbar_label, 'shrink': 0.8},
        ax=ax,
        annot_kws={'size': 8}
    )

    ax.set_title(title or default_title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Amino Acid', fontsize=12)
    ax.set_ylabel('Amino Acid', fontsize=12)
    ax.tick_params(axis='both', labelsize=10)
    plt.setp(ax.get_xticklabels(), rotation=0)
    plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved pairwise matrix to: {save_path}")

        if save_text_report and metric == 'accuracy':
            save_pairwise_report(
                matrix_df, pair_details,
                _report_path(save_path),
                title or default_title
            )

    plt.close()
    return matrix_df


def plot_pairwise_confusion_summary(
    results_dict: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5),
    top_n_confusions: int = 15,
    save_text_report: bool = True
) -> Dict[str, Any]:
    """Create a summary visualization showing most confused amino acid pairs."""
    amino_acids = _discover_amino_acids(results_dict)

    acc_matrix, pair_details = compute_pairwise_classification_accuracy(
        results_dict, None, amino_acids
    )

    pair_accuracies = []
    for (aa1, aa2), details in pair_details.items():
        pair_accuracies.append({
            'pair': f'{aa1}-{aa2}',
            'accuracy': details['accuracy'],
            'total': details['total']
        })

    df_pairs = pd.DataFrame(pair_accuracies).sort_values('accuracy')

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    worst_pairs = df_pairs.head(top_n_confusions)
    colors = plt.cm.RdYlGn(worst_pairs['accuracy'].values)

    axes[0].barh(worst_pairs['pair'], worst_pairs['accuracy'], color=colors,
                 edgecolor='black', linewidth=0.5)
    axes[0].set_xlabel('Pairwise Accuracy', fontsize=11)
    axes[0].set_ylabel('Amino Acid Pair', fontsize=11)
    axes[0].set_title(f'Most Confused Pairs (Top {top_n_confusions})', fontsize=12, fontweight='bold')
    axes[0].set_xlim(0, 1)
    axes[0].axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random Chance')
    axes[0].legend(loc='lower right', fontsize=9)
    axes[0].grid(axis='x', alpha=0.3)

    for idx, (_, row) in enumerate(worst_pairs.iterrows()):
        axes[0].text(row['accuracy'] + 0.02, idx, f"{row['accuracy']:.2f}",
                     va='center', fontsize=8)

    acc_values = df_pairs['accuracy'].dropna()
    axes[1].hist(acc_values, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    axes[1].axvline(x=acc_values.mean(), color='red', linestyle='-', linewidth=2,
                    label=f'Mean: {acc_values.mean():.3f}')
    axes[1].axvline(x=acc_values.median(), color='orange', linestyle='--', linewidth=2,
                    label=f'Median: {acc_values.median():.3f}')
    axes[1].axvline(x=0.5, color='gray', linestyle=':', linewidth=1.5, label='Random Chance')
    axes[1].set_xlabel('Pairwise Accuracy', fontsize=11)
    axes[1].set_ylabel('Number of Pairs', fontsize=11)
    axes[1].set_title('Distribution of Pairwise Accuracies', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper left', fontsize=9)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved pairwise summary to: {save_path}")

        if save_text_report:
            save_pairwise_report(
                acc_matrix, pair_details,
                _report_path(save_path),
                "Pairwise Confusion Summary"
            )

    plt.close()

    return {
        'total_pairs': len(df_pairs),
        'mean_accuracy': acc_values.mean(),
        'median_accuracy': acc_values.median(),
        'std_accuracy': acc_values.std(),
        'min_accuracy': acc_values.min(),
        'max_accuracy': acc_values.max(),
        'pairs_below_random': (acc_values < 0.5).sum(),
        'pairs_below_60': (acc_values < 0.6).sum(),
        'pairs_below_70': (acc_values < 0.7).sum(),
        'pairs_above_90': (acc_values >= 0.9).sum(),
        'pairs_above_95': (acc_values >= 0.95).sum(),
        'worst_pairs': worst_pairs[['pair', 'accuracy', 'total']].to_dict('records'),
        'best_pairs': df_pairs.tail(top_n_confusions)[['pair', 'accuracy', 'total']].to_dict('records'),
        'matrix': acc_matrix,
        'pair_details': pair_details
    }


def plot_category_pairwise_matrix(
    results_dict: Dict[str, Dict[str, Any]],
    classification_mode: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    save_text_report: bool = True
) -> pd.DataFrame:
    """Create pairwise classification matrix at the category level."""
    from vrhmm.utils.amino_acids import get_amino_acid_category, get_all_categories

    categories = get_all_categories(classification_mode)

    category_signals: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for result in results_dict.values():
        aa = result.get('amino_acid', '')
        if aa:
            try:
                cat = get_amino_acid_category(aa, classification_mode)
                category_signals[cat].append(result)
            except ValueError:
                continue

    n_cat = len(categories)
    accuracy_matrix = np.full((n_cat, n_cat), np.nan)
    pair_details = {}

    for i, cat1 in enumerate(categories):
        accuracy_matrix[i, i] = 1.0
        for j, cat2 in enumerate(categories):
            if i >= j:
                continue

            signals_cat1 = category_signals.get(cat1, [])
            signals_cat2 = category_signals.get(cat2, [])

            if not signals_cat1 or not signals_cat2:
                continue

            correct = 0
            total = 0

            for result in signals_cat1:
                pred = result.get('predicted_category', '')
                if pred in (cat1, cat2):
                    if pred == cat1:
                        correct += 1
                    total += 1

            for result in signals_cat2:
                pred = result.get('predicted_category', '')
                if pred in (cat1, cat2):
                    if pred == cat2:
                        correct += 1
                    total += 1

            if total > 0:
                acc = correct / total
                accuracy_matrix[i, j] = acc
                accuracy_matrix[j, i] = acc
                pair_details[(cat1, cat2)] = {
                    'accuracy': acc,
                    'total': total,
                    'correct': correct
                }

    matrix_df = pd.DataFrame(accuracy_matrix, index=categories, columns=categories)

    mask = np.triu(np.ones_like(matrix_df, dtype=bool), k=0)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        matrix_df,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        vmin=0.5,
        vmax=1.0,
        square=True,
        linewidths=1,
        linecolor='white',
        cbar_kws={'label': 'Accuracy', 'shrink': 0.8},
        ax=ax,
        annot_kws={'size': 11, 'weight': 'bold'}
    )

    ax.set_title(f'Pairwise Category Classification ({classification_mode})',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Category', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved category pairwise matrix to: {save_path}")

        if save_text_report:
            save_pairwise_report(
                matrix_df, pair_details,
                _report_path(save_path),
                f"Pairwise Category Classification ({classification_mode})"
            )

    plt.close()
    return matrix_df
    