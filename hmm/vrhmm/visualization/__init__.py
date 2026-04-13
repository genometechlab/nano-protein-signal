"""Visualization functionality."""

from vrhmm.visualization.classification_plots import generate_classification_report
from vrhmm.visualization.classification_plots import (
    generate_classification_report,
    plot_confusion_matrix,
    plot_category_performance,
    save_text_report
)

__all__ = [
    "generate_classification_report",
    "plot_hmm_segmentation_and_path",
    "plot_multi_panel_hmm_states",
    "plot_segment_pileup",
    "plot_match_state_pileup",
    "plot_backslip_distribution",
    "plot_skip_distribution",
    "plot_backslip_by_position",
    "plot_segmentation_only"
]