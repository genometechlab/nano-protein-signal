"""Classification visualization utilities."""

import logging
from pathlib import Path
from typing import Optional, Tuple

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
    """Generate all classification visualizations and save to output_dir."""
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

    ax.set_title(
        f'{mode.upper()} Classification\n'
        f'Accuracy: {accuracy:.1f}% | Balanced Accuracy: {balanced_acc:.1f}%'
    )
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

    df_metrics = pd.DataFrame(metrics).sort_values('Accuracy')

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(df_metrics['Category'], df_metrics['Accuracy'])

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
        