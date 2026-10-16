"""Figures and path-statistics tables for the HMM cross-validation (Figs 6-7, Supp. Figs 10-12, Supp. Tables S8-S11).

Reads the outputs of hmm/hmm_cv_evaluation.py (--cv-dir) and, for the Fig. 7 / Supp. Fig. 12 signal, the Dynp-34
segment pickle (--segments). Data: the primary analysis (Dynp-34, re-estimated transitions), pooled out-of-fold over the
in-bounds held-out regions; each region is scored by the models of the fold that held it out.

Writes under --out-dir:
  figures/<name>.pdf, .svg and .png (600 dpi), one file set per panel:
    fig6_confusion_{2way,3way,polarity}   confusion matrices, counts and row %
    fig6_traceback_{H,V,P}                Viterbi tracebacks: H vs H/D, V vs V/H/D, P vs P/H/D/T
    suppfig10_pairwise_accuracy           pairwise HMM accuracy matrix
    suppfig11_confusion_5way_size, suppfig11_traceback_W (W vs W/E/H/S/D)
    fig6_composite, suppfig11_composite   the assembled figures
    fig7_hmm_states                       Fig. 7: state-coloured signal, A = G region, B = D region
    suppfig12_hmm_states                  Supp. Fig. 12: state-coloured signal, P region
  tables/path_stats_{2way,3way,polarity,5way_size}.csv   HMM path statistics per class (Supp. Tables S8-S11)
  tables/pairwise_accuracy_matrix.csv
  FIGURES.md   every file with a one-line description, and the log-probabilities shown in each traceback panel.

Plotting code
  Confusion matrices: vrhmm.visualization.classification_plots.plot_confusion_matrix, adapted to the published panels
    (see plot_confusion_matrix below for every change).
  Tracebacks: plot_traceback below, which recreates the published panels (results/04_14_26_ALL_IQR_TRACES_35_SEGS_
    MODEL_TESTS/plots/*_plots.pdf, whose plotting code is not in the repository). Colours, line styles and widths,
    markers, alphas, font sizes and the legend were read from those PDFs' drawing operators. Each emitting state of a
    model's Viterbi path (vrhmm full_path) is a point (match-state index, segment number); Insert_i_j sits at x = i.
    Unlike the published panels, each model's line is offset horizontally by a fixed amount (MODEL_OFFSETS, within
    +-0.15 match states; x COMPOSITE_OFFSET_SCALE = 3, within +-0.45, in the composites) so overlapping paths stay
    visible; the legend title says so.
  Pairwise matrix: vrhmm.visualization.pairwise_matrix.plot_pairwise_classification_matrix, unchanged (RdYlGn, 0.5-1.0,
    lower triangle), from the 20 held-out per-class scores.
  State-coloured signal (Fig. 7, Supp. Fig. 12): plot_state_signal below, which recreates the published panels
    (results/04_13_26_ALL_IQR_TRACES_35_SEGS/plots/*_backslip.pdf as edited into the manuscript; no plotting code in the
    repository). The region's 35 Dynp segments (cleaned_segments) are concatenated and z-normalised over the region
    (vrhmm SignalProcessor._z_normalize_signal); each segment is drawn in the colour of the state that emitted it on
    the region's Viterbi path under its own amino-acid model (held-out fold), labelled with the match-state number at
    the segment's midpoint just above the trace. Dashed red / dashed blue lines mark segments entered by a skip / a
    backslip (slip), a blue band marks each post-backslip segment, and a dotted green line marks each inserted segment
    (drawn in light green). Colours, line styles, widths and alphas were read from the published PDFs.
  Path statistics: parse_path_stats and build_path_stats_table from hmm/vrhmm/HMM_Plots_and_Data_Processing.ipynb
    (copied below), on each region's Viterbi path under its best-scoring (20-way predicted) model, as vrhmm's full_path.

Traceback example rule (fixed before plotting): for a comparison of class c against models M, margin = ll_c - max over
the other models in M of ll_m (forward log-likelihoods). The example is the held-out region of class c whose margin is
closest to the median margin of all held-out regions of class c; ties go to the smallest df_index.

Fig. 7 / Supp. Fig. 12 regions (STATE_FIGS): the published regions where they are held out and still show the
described event (Fig. 7A G 955; Supp. Fig. 12 P 6009). Otherwise (Fig. 7B) the fallback rule: among the held-out
regions of that amino acid with at least one slip call on the Viterbi path under their own model, the one whose
own-class margin (ll_own - max over the other 19 models) is closest to the median margin of all held-out regions of
that amino acid; ties go to the smallest df_index.

Usage (from the repository root):
  python hmm/hmm_figures.py --cv-dir results/hmm_cv --out-dir results/hmm_figures
(--segments defaults to the Dynp-34 pickle in hmm/vrhmm/data/, as in hmm_cv_evaluation.py)
"""
import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, confusion_matrix  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "hmm"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from vrhmm.processing.signal_processor import SignalProcessor  # noqa: E402
from vrhmm.utils.amino_acids import get_classification_mode  # noqa: E402
from vrhmm.visualization.pairwise_matrix import plot_pairwise_classification_matrix  # noqa: E402

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

AAS = list("ACDEFGHIKLMNPQRSTVWY")
PRIMARY = "HMM primary (Dynp-34, re-estimated transitions) | in-bounds"
VRHMM_MODE = {"2way": "2way", "3way": "3way", "polarity": "biological", "5way_size": "5way_size"}
CONFUSION = {"fig6_confusion_2way": "2way", "fig6_confusion_3way": "3way", "fig6_confusion_polarity": "polarity",
             "suppfig11_confusion_5way_size": "5way_size"}
# figure: (true class, models, model-label format as in the published figure)
TRACEBACKS = {"fig6_traceback_H": ("H", ["H", "D"], "{c} vs {m}-model"),
              "fig6_traceback_V": ("V", ["V", "H", "D"], "{c} vs {m} Model"),
              "fig6_traceback_P": ("P", ["P", "H", "D", "T"], "{c} vs {m} Model"),
              "suppfig11_traceback_W": ("W", ["W", "E", "H", "S", "D"], "{c} vs {m} Model")}
N_STATES = 35


def save(fig, out, name, tight=True):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight" if tight else None)
    plt.close(fig)
    return [f"figures/{name}.{ext}" for ext in ("pdf", "svg", "png")]


# ── Confusion matrices ────────────────────────────────────────────────

def plot_confusion_matrix(df, mode, categories, majority_acc):
    """vrhmm.visualization.classification_plots.plot_confusion_matrix, changed to match the published panels
    (results/04_13_26_ALL_IQR_TRACES_35_SEGS/plots/confusion_matrix_*.pdf):
      - categories in vrhmm's category order (get_classification_mode), not sorted;
      - cells coloured by row-normalised recall (imshow, Blues, 0-1) instead of a seaborn heatmap of counts;
      - each nonzero cell annotated 'count (row %)' in bold, white on cells above 0.5; empty cells left blank;
      - white cell borders; colour bar labelled 'Recall (row-normalized)';
      - per-row 'recall % (n)' labels in grey to the right of the matrix;
      - title 'Confusion Matrix — <task> (Accuracy: x%, n=N)'; bold axis labels; x labels at 45 degrees.
    Beyond the published look:
      - a second title line with balanced accuracy and the training-fold majority-class accuracy, since every HMM
        result is reported next to the majority baseline;
      - the per-row labels sit between the matrix and the colour bar instead of overlapping it;
      - the figure is returned and saved as PDF, SVG and 600-dpi PNG instead of one 150-dpi PDF.
    """
    cm = confusion_matrix(df["true_category"], df["predicted_category"], labels=categories)
    rs = cm.sum(1, keepdims=True)
    rn = np.divide(cm, rs, out=np.zeros(cm.shape), where=rs > 0)
    k = len(categories)
    fig, ax = plt.subplots(figsize=(1.15 * k + 3.4, 1.05 * k + 2.6))
    im = ax.imshow(rn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(-0.5, k, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, k, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    for i in range(k):
        for j in range(k):
            if cm[i, j] > 0:
                ax.text(j, i, f"{cm[i, j]}\n({100 * rn[i, j]:.0f}%)", ha="center", va="center", fontsize=11,
                        fontweight="bold", color="white" if rn[i, j] > 0.5 else "black")
        ax.text(k - 0.5 + 0.12, i, f"{100 * rn[i, i]:.0f}% ({int(rs[i, 0])})", ha="left", va="center", fontsize=11,
                color="gray", clip_on=False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.2)
    cbar.set_label("Recall (row-normalized)", fontsize=11)
    ax.set_xticks(range(k), categories, rotation=45, ha="right", fontsize=11)
    ax.set_yticks(range(k), categories, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=14, fontweight="bold")
    ax.set_ylabel("True", fontsize=14, fontweight="bold")
    acc = (df["true_category"] == df["predicted_category"]).mean() * 100
    bal = balanced_accuracy_score(df["true_category"], df["predicted_category"]) * 100
    ax.set_title(f"Confusion Matrix — {mode} (Accuracy: {acc:.1f}%, n={len(df)})\n"
                 f"Balanced accuracy: {bal:.1f}% | Majority-class accuracy: {100 * majority_acc:.1f}%",
                 fontsize=13, fontweight="bold")
    return fig, cm, acc, bal


# ── Path statistics (from hmm/vrhmm/HMM_Plots_and_Data_Processing.ipynb, cell 0 and cell 3) ──

CLASSIFICATION_MODES = {
    '2way': {'positive': ['K', 'R', 'H'], 'negative': ['D', 'E']},
    '3way': {'positive': ['K', 'R', 'H'], 'negative': ['D', 'E'],
             'neutral': ['S', 'T', 'N', 'Q', 'C', 'G', 'A', 'V', 'L', 'I', 'M', 'P', 'F', 'W', 'Y']},
    '5way_size': {'very_small': ['G', 'A', 'S'], 'small': ['C', 'D', 'N', 'P', 'T', 'V'], 'medium': ['E', 'I', 'L', 'Q'],
                  'large': ['H', 'K', 'M', 'F'], 'very_large': ['R', 'W', 'Y']},
    'polarity': {'nonpolar': ['A', 'G', 'I', 'L', 'M', 'F', 'W', 'P', 'V'], 'polar': ['S', 'T', 'N', 'Q', 'C', 'Y'],
                 'positive': ['K', 'R', 'H'], 'negative': ['D', 'E']},
}
CATEGORY_ORDERS = {
    '2way': ['positive', 'negative'],
    '3way': ['positive', 'negative', 'neutral'],
    'polarity': ['nonpolar', 'polar', 'positive', 'negative'],
    '5way_size': ['very_small', 'small', 'medium', 'large', 'very_large'],
}


def get_category(amino_acid, mode):
    if mode == '20way':
        return amino_acid
    for cat, aa_list in CLASSIFICATION_MODES[mode].items():
        if amino_acid in aa_list:
            return cat
    raise ValueError(f"{amino_acid} not found in {mode}")


def parse_path_stats(full_path):
    stats = {'n_matches': 0, 'n_skips': 0, 'n_backslips': 0, 'n_self_loops': 0, 'n_inserts': 0, 'n_unique_matches': 0,
             'n_total_path_states': len(full_path)}
    prev_match_idx = -1
    prev_state = None
    match_indices = []
    for state in full_path:
        if state.startswith('Match_'):
            idx = int(state.split('_')[1])
            match_indices.append(idx)
            if idx == prev_match_idx and state == prev_state:
                stats['n_self_loops'] += 1
            else:
                stats['n_matches'] += 1
            prev_match_idx = idx
            prev_state = state
        elif state.startswith('Insert'):
            stats['n_inserts'] += 1
        elif state.startswith('Skip'):
            stats['n_skips'] += 1
        elif state.startswith('Slip'):
            stats['n_backslips'] += 1
    stats['n_unique_matches'] = len(set(match_indices))
    return stats


def build_path_stats_table(results_dict, mode='20way', save_path=None):
    category_order = CATEGORY_ORDERS.get(mode, [])
    cat_stats = defaultdict(list)
    for trace_id, result in results_dict.items():
        true_aa = result.get('amino_acid')
        full_path = result.get('full_path', [])
        if true_aa is None or not full_path:
            continue
        try:
            true_cat = get_category(true_aa, mode)
        except ValueError:
            continue
        cat_stats[true_cat].append(parse_path_stats(full_path))
    rows = []
    stat_keys = ['n_unique_matches', 'n_backslips', 'n_skips', 'n_self_loops', 'n_inserts']
    for cat in category_order:
        if cat not in cat_stats:
            continue
        traces = cat_stats[cat]
        row = {'Category': cat, 'n_traces': len(traces)}
        for key in stat_keys:
            values = [t[key] for t in traces]
            row[f'{key}_mean'] = np.mean(values)
            row[f'{key}_std'] = np.std(values)
        rows.append(row)
    df = pd.DataFrame(rows)
    if save_path:
        df.to_csv(save_path, index=False)
    return df


# ── Tracebacks (published style) ──────────────────────────────────────

# per model, in comparison order: colour, line style, line width, marker, alpha (published PDFs)
MODEL_STYLES = [("tab:green", "-", 2.5, "o", 0.85), ("tab:red", "--", 2, "s", 0.8), ("tab:blue", "-.", 2, "D", 0.75),
                ("tab:orange", ":", 2, "^", 0.7), ("tab:purple", "-", 2, "v", 0.65)]
# horizontal offset (match states) per model slot, the same in every panel, so overlapping paths stay visible;
# two models sit at -/+0.15 and five spread evenly over -0.15..+0.15
MODEL_OFFSETS = (-0.15, 0.15, 0.0, -0.075, 0.075)
OFFSET_NOTE = "(lines offset slightly for visibility)"
# overlays on the steps into a point: colour, line style, line width (alpha 0.9)
EVENT_STYLES = {"skip": ("tab:red", "--", 3), "slip": ("tab:blue", "--", 3), "insert": ("tab:orange", ":", 2.5),
                "self_loop": ("magenta", "-", 2.5)}


def traceback_points(path):
    """Emitting states of a Viterbi path as (match-state index, segment number, step type into the point)."""
    pts, prev = [], None
    for s in path:
        if not (s.startswith("Match_") or s.startswith("Insert_")):
            continue                                  # skip and slip states emit nothing
        x = int(s.split("_")[1])
        if prev is None:
            kind = "start"
        elif s.startswith("Insert_"):
            kind = "insert"
        elif s == prev[1]:
            kind = "self_loop"
        elif x > prev[0] + 1:
            kind = "skip"
        elif x < prev[0]:
            kind = "slip"
        else:
            kind = "match"
        pts.append((x, len(pts), kind))
        prev = (x, s)
    return pts


def plot_traceback(ax, paths, n_states=N_STATES, s=1.0, fs_tick=8, fs_label=10, offset_scale=1.0):
    """One line per model in (match-state index, segment number); skip and slip steps get a dashed overlay and an
    arrow in the model's colour, inserts a dotted and self-loops a solid vertical overlay. Each model's line, overlays
    and arrows are shifted horizontally by its MODEL_OFFSETS entry times offset_scale. s scales line widths and marker
    sizes (1 = the published single panel)."""
    ax.plot([0, n_states], [0, n_states], color="gray", ls=":", lw=1 * s, alpha=0.4, zorder=1)
    arrows = []
    for (c, ls, lw, mk, a), off, path in zip(MODEL_STYLES, MODEL_OFFSETS, paths):
        pts = [(x + off * offset_scale, y, kind) for x, y, kind in traceback_points(path)]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, ls=ls, lw=lw * s, marker=mk, ms=4 * s, mec="white",
                mew=0.5 * s, alpha=a)
        for (x0, y0, _), (x1, y1, kind) in zip(pts[:-1], pts[1:]):
            if kind in EVENT_STYLES:
                ec, els, elw = EVENT_STYLES[kind]
                ax.plot([x0, x1], [y0, y1], color=ec, ls=els, lw=elw * s, alpha=0.9)
            if kind in ("skip", "slip"):
                arrows.append(((x0, y0), (x1, y1), c))
    for start, end, c in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color=c, lw=2.5 * s, mutation_scale=10 * s))
    ax.set_xlim(-0.5, n_states + 0.5)
    ax.set_ylim(-0.5, n_states + 0.5)
    ax.set_xticks(range(0, n_states + 1, 5))
    ax.set_yticks(range(0, n_states + 1, 5))
    ax.tick_params(labelsize=fs_tick)
    ax.grid(True, alpha=0.15, linewidth=0.5 * s)
    ax.set_xlabel("Match State Index", fontsize=fs_label, fontweight="bold")
    ax.set_ylabel("Segment Number", fontsize=fs_label, fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(1.5 * s)


def path_kinds(paths):
    """Step types drawn in a traceback panel (skip, slip, insert, self_loop)."""
    return {kind for path in paths for _, _, kind in traceback_points(path)}


def traceback_legend(fig, model_labels, anchor, kinds):
    """The published legend: model entries, one blank slot, then the path-event key, in 5 columns. Only what is drawn
    is listed: event entries (and the skip/slip arrows) for the step types in the panel (kinds); the published Start and
    End entries are left out because no start or end markers are drawn."""
    handles = [Line2D([0], [0], color=c, ls=ls, lw=2.5, marker=mk, ms=5) for c, ls, _, mk, _ in MODEL_STYLES[:len(model_labels)]]
    labels = list(model_labels)
    key = [(None, Line2D([], [], color="none"), " "), (None, Line2D([0], [0], color="tab:green", lw=2), "Match (diagonal)"),
           ("skip", Line2D([0], [0], color="tab:red", ls="--", lw=2.5), "Skip (forward jump \u2192)"),
           ("slip", Line2D([0], [0], color="tab:blue", ls="--", lw=2.5), "Slip (backward jump \u2190)"),
           ("self_loop", Line2D([0], [0], color="magenta", lw=2), "Self-loop (vertical)"),
           ("insert", Line2D([0], [0], color="tab:orange", ls=":", lw=2), "Insert (vertical)"),
           ("skip", Line2D([], [], color="tab:red", marker=">", ls="none", ms=10), "Skip arrow (\u2192)"),
           ("slip", Line2D([], [], color="tab:blue", marker="<", ls="none", ms=10), "Slip arrow (\u2190)"),
           (None, Line2D([0], [0], color="gray", ls=":", lw=1), "Ideal diagonal")]
    for kind, h, lab in key:
        if kind is None or kind in kinds:
            handles.append(h)
            labels.append(lab)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=anchor, ncol=5, fontsize=8, title=OFFSET_NOTE,
               title_fontsize=8)


# ── Composite figures (layout measured from the revision manuscript and supplement PDFs) ──

# Figure 6 (manuscript p. 20) and Supp. Fig. 11 (supplement p. 12), in inches from the figure's top-left corner:
# overall size; per row: confusion-matrix and traceback axes as (left, top offset in the row, width, height)
FIG6_SIZE, FIG6_ROW_TOPS = (6.30, 7.07), (0.0, 2.33, 4.66)
FIG6_CM, FIG6_TB = (0.63, 0.09, 2.04, 2.00), (3.22, 0.12, 2.91, 1.88)
S11_SIZE, S11_CM, S11_TB = (6.45, 2.25), (0.66, 0.04, 2.08, 2.00), (3.30, 0.04, 3.02, 1.98)
TRUE_LABEL_X = 0.14             # right edge of the rotated 'True' label, inches from the figure's left edge
COMPOSITE_SCALE = 0.7           # traceback line widths and markers relative to the single panel
COMPOSITE_OFFSET_SCALE = 3      # traceback offsets relative to the single panel (+-0.45 match states)


def add_axes_in(fig, box, row_top=0.0):
    W, H = fig.get_size_inches()
    left, top, w, h = box
    return fig.add_axes([left / W, 1 - (row_top + top + h) / H, w / W, h / H])


def display_name(cat):
    return cat.replace("non_polar", "nonpolar").replace("_", " ")


def draw_confusion_compact(ax, g, cats, ytick_rotation=0, xlabel=True, fs=6.5, label_x=None):
    """Confusion matrix as in the published Figure 6: row-normalised Blues, 'count (row %)' in every cell (zeros
    included), white cell borders, no colour bar or title."""
    cm = confusion_matrix(g["true_category"], g["predicted_category"], labels=cats)
    rs = cm.sum(1, keepdims=True)
    rn = np.divide(cm, rs, out=np.zeros(cm.shape), where=rs > 0)
    k = len(cats)
    ax.imshow(rn, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(-0.5, k, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, k, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{cm[i, j]}\n({100 * rn[i, j]:.0f}%)", ha="center", va="center", fontsize=fs,
                    color="white" if rn[i, j] > 0.5 else "black")
    names = [display_name(c) for c in cats]
    ax.set_xticks(range(k), names, fontsize=fs)
    ax.set_yticks(range(k), names, fontsize=fs, rotation=ytick_rotation, va="center")
    ax.tick_params(which="major", length=2, pad=1.5)
    ax.set_ylabel("True", fontsize=9, fontweight="bold", labelpad=2)
    if label_x is not None:
        ax.yaxis.set_label_coords(label_x, 0.5)
    if xlabel:
        ax.set_xlabel("Predicted", fontsize=9, fontweight="bold", labelpad=2)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)


def draw_traceback_compact(ax, paths, model_labels, legend, loc, xlabel=True):
    """The traceback at composite size, with the published composite's in-axes legend: 'full' (models and the
    path-event key, listing only the step types drawn in the panel), 'ideal' (models and the ideal alignment) or
    'models'."""
    s = COMPOSITE_SCALE
    plot_traceback(ax, paths, s=s, fs_tick=7, fs_label=9, offset_scale=COMPOSITE_OFFSET_SCALE)
    if not xlabel:
        ax.set_xlabel("")
    handles = [Line2D([0], [0], color=c, ls=ls, lw=2.5 * s, marker=mk, ms=5 * s)
               for c, ls, _, mk, _ in MODEL_STYLES[:len(model_labels)]]
    labels = list(model_labels)
    if legend == "full":
        kinds = path_kinds(paths)
        key = [(None, Line2D([0], [0], color="tab:green", lw=2 * s), "Match (diagonal)"),
               ("skip", Line2D([0], [0], color="tab:red", ls="--", lw=2.5 * s), "Skip (forward jump)"),
               ("slip", Line2D([0], [0], color="tab:blue", ls="--", lw=2.5 * s), "Slip (backward jump)"),
               ("self_loop", Line2D([0], [0], color="magenta", lw=2 * s), "Self-loop (vertical)"),
               ("insert", Line2D([0], [0], color="tab:orange", ls=":", lw=2 * s), "Insert (vertical)")]
        for kind, h, lab in key:
            if kind is None or kind in kinds:
                handles.append(h)
                labels.append(lab)
    if legend in ("full", "ideal"):
        handles.append(Line2D([0], [0], color="gray", ls=":", lw=1))
        labels.append("Ideal Alignment")
    ax.legend(handles, labels, loc=loc, fontsize=6, handlelength=2.2, labelspacing=0.3, borderpad=0.4, title=OFFSET_NOTE,
              title_fontsize=6)


def pick_example(scores, cls, models):
    g = scores[scores.aa == cls].copy()
    others = [f"ll_{m}" for m in models if m != cls]
    g["margin"] = g[f"ll_{cls}"] - g[others].max(axis=1)
    med = g.margin.median()
    g["dist"] = (g.margin - med).abs()
    best = g.sort_values(["dist", "df_index"]).iloc[0]
    return best, float(med), len(g)


# ── State-coloured signal (Fig. 7, Supp. Fig. 12; published style) ─────

# Fig. 7 and Supp. Fig. 12: figure -> [(panel letter, amino acid, df_index or None for the fallback rule)]. G 955 and
# P 6009 are the published regions (held out, event still shown). The published D region (3547) is replaced by the
# fallback rule because the published Fig. 7B caption does not describe a skip.
STATE_FIGS = {"fig7_hmm_states": [("A", "G", 955), ("B", "D", None)], "suppfig12_hmm_states": [(None, "P", 6009)]}
# figure size and the axes box (left, top, width, height) of each panel, inches (manuscript p. 22, supplement p. 12)
STATE_SIZES = {"fig7_hmm_states": ((6.5, 2.9), [(0.42, 0.12, 6.0, 1.25), (0.42, 1.57, 6.0, 1.25)]),
               "suppfig12_hmm_states": ((6.45, 1.3), [(0.42, 0.05, 5.95, 1.2)])}
# match-state colours 0-34 (published label colours)
STATE_COLORS = ["#8b0000", "#006400", "#00008b", "#4b0082", "#2f4f4f", "#8b4513", "#800080", "#008080", "#191970",
                "#556b2f", "#8b008b", "#008b8b", "#b8860b", "#a52a2a", "#483d8b", "#2e8b57", "#800000", "#000080",
                "#696969", "#d2691e", "#9932cc", "#1a1a1a", "#4682b4", "#5f9ea0", "#b22222", "#2b0b3f", "#0b203f",
                "#0b3f1b", "#3f2b0b", "#3f0b0b", "#1c3144", "#3f0071", "#590000", "#004000", "#400040"]
INSERT_STYLE = dict(color="lime", alpha=0.7)
BOUNDARY_STYLES = {"skip": dict(color="red", ls="--", lw=0.75, alpha=0.9),
                   "slip": dict(color="blue", ls="--", lw=0.75, alpha=0.9),
                   "insert": dict(color="green", ls=":", lw=1.5, alpha=0.8)}
POST_SLIP_BAND = dict(color="blue", alpha=0.35, lw=0, ymin=0.92, ymax=1.0)


def path_steps(path):
    """Emitting states of a Viterbi path, each with the events on the way into it: [(state, previous emitting state,
    {'skip', 'slip', 'insert'})]. A run of Slip states between two emitting states is one backslip."""
    steps, silent, prev = [], [], None
    for st in path:
        if st.startswith(("Skip_", "Slip_")):
            silent.append(st)
            continue
        ev = {k for k, pre in (("skip", "Skip_"), ("slip", "Slip_")) if any(x.startswith(pre) for x in silent)}
        if st.startswith("Insert_"):
            ev.add("insert")
        steps.append((st, prev, ev))
        silent, prev = [], st
    return steps


def state_label(st):
    return st.split("_", 1)[1].replace("_", "-") if st.startswith("Insert_") else st.split("_")[1]


def pick_slip_example(scores, paths, cls):
    """Fallback rule: held-out regions of cls with >= 1 slip call under the own model; margin closest to the class
    median (median over all held-out regions of cls); ties go to the smallest df_index."""
    g = scores[scores.aa == cls].copy()
    g["margin"] = g[f"ll_{cls}"] - g[[f"ll_{a}" for a in AAS if a != cls]].max(axis=1)
    med = g.margin.median()
    own = paths[paths.model == cls].set_index("df_index").path
    g["slips"] = g.df_index.map(lambda k: own[k].count("Slip_"))
    el = g[g.slips > 0].assign(dist=lambda d: (d.margin - med).abs()).sort_values(["dist", "df_index"])
    return int(el.df_index.iloc[0]), float(med), len(g), len(el)


def plot_state_signal(ax, segs, path, legend=False):
    """One region's z-normalised signal, each Dynp segment coloured by the state that emitted it on the Viterbi path;
    returns the event list [(segment, sample, previous state, state, events)]."""
    z = SignalProcessor._z_normalize_signal(np.concatenate(segs))
    steps = path_steps(path)
    if len(steps) != len(segs):
        raise ValueError(f"{len(steps)} emitting states for {len(segs)} segments")
    events, start = [], 0
    for j, (seg, (st, prev, ev)) in enumerate(zip(segs, steps)):
        x = np.arange(start, min(start + len(seg) + 1, len(z)))   # one sample of overlap joins adjacent segments
        y = z[x]
        if st.startswith("Insert_"):
            ax.plot(x, y, lw=0.5, **INSERT_STYLE)
        else:
            k = int(st.split("_")[1])
            ax.plot(x, y, color=STATE_COLORS[k], lw=0.5)
            ax.text(x[len(x) // 2], y.max() + 0.1, str(k), color=STATE_COLORS[k], fontsize=6, ha="center", va="bottom")
        for kind in ("skip", "slip", "insert"):
            if kind in ev:
                ax.axvline(start, **BOUNDARY_STYLES[kind])
        if "slip" in ev:
            ax.axvspan(start, start + len(seg), **POST_SLIP_BAND)
        if ev:
            events.append((j, start, prev, st, ev))
        start += len(seg)
    ax.set_xlim(0, len(z))
    ax.set_ylim(min(-3.0, z.min() - 0.1), max(2.5, z.max() + 1.0))
    ax.set_yticks(range(int(np.ceil(ax.get_ylim()[0])), int(np.floor(ax.get_ylim()[1])) + 1))
    ax.set_xticks([])
    ax.tick_params(labelsize=6, length=2, pad=1.5)
    ax.set_ylabel("Z-Normalized Current", fontsize=7, labelpad=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)
    if legend:                                        # the published legend, listing only what is drawn in this panel
        drawn = set().union(*(e[4] for e in events))
        key = [("insert", Patch(**INSERT_STYLE), "Insert"),
               ("insert", Line2D([0], [0], **BOUNDARY_STYLES["insert"]), "Insert Boundary"),
               ("skip", Line2D([0], [0], **BOUNDARY_STYLES["skip"]), "Skip Boundary"),
               ("slip", Line2D([0], [0], **BOUNDARY_STYLES["slip"]), "Backslip Boundary"),
               ("slip", Patch(color="blue", alpha=0.35, lw=0), "Post-backslip Segment")]
        key = [(h, lab) for kind, h, lab in key if kind in drawn]
        if key:
            ax.legend(*zip(*key), loc="lower center", fontsize=5.5, handlelength=2, labelspacing=0.2, borderpad=0.3,
                      framealpha=0.8)
    return events


def describe_event(prev, st, ev):
    """Plain-language event: backslip a->b, skip a->b (states skipped), insert between a and b."""
    out = []
    a = state_label(prev) if prev else "start"
    b = state_label(st)
    if "slip" in ev:
        out.append(f"backslip {a}\u2192{b}")
    if "skip" in ev:
        lo = int(a.split("-")[-1]) + (0 if prev.startswith("Insert_") else 1)
        hi = int(b.split("-")[0]) - (0 if st.startswith("Insert_") else 1)
        out.append(f"skip {a}\u2192{b} (states {lo}\u2013{hi} skipped)" if hi > lo else f"skip {a}\u2192{b} (state {lo} skipped)")
    if "insert" in ev:
        out.append(f"insert between states {b.replace('-', ' and ')}")
    return "; ".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cv-dir", default="results/hmm_cv", help="output directory of hmm_cv_evaluation.py")
    ap.add_argument("--out-dir", default="results/hmm_figures")
    ap.add_argument("--segments", default="hmm/vrhmm/data/denoise_segment_dynp_parallel_34_BkP_full_pastor_norm_features_"
                    "clean_1751_4570_min_size_30.pkl", help="Dynp-34 segment pickle (signal for Fig. 7 / Supp. Fig. 12)")
    args = ap.parse_args()
    cv, out = Path(args.cv_dir), Path(args.out_dir)
    figdir, tabdir = out / "figures", out / "tables"
    figdir.mkdir(parents=True, exist_ok=True)
    tabdir.mkdir(parents=True, exist_ok=True)

    reg = pd.read_csv(cv / "regions.csv").set_index("df_index")
    scores = pd.read_csv(cv / "scores.csv.gz")
    scores = scores[(scores.analysis == "primary") & (scores.transitions == "reestimated")].reset_index(drop=True)
    preds = pd.read_csv(cv / "predictions.csv.gz")
    preds = preds[preds.analysis == "primary"]
    summary = pd.read_csv(cv / "summary.csv")
    paths = pd.read_csv(cv / "viterbi_paths.csv.gz", keep_default_na=False)
    key = {k: f"{r.run}_{r.channel}_{k}_{r.aa}" for k, r in reg.iterrows()}     # vrhmm results-key format
    md = ["# HMM figures and tables", "",
          f"Generated by `hmm/hmm_figures.py` from `{cv}` (`hmm/hmm_cv_evaluation.py`). Data: the primary analysis "
          f"(Dynp-34, re-estimated transitions), pooled out-of-fold over the {len(scores)} in-bounds held-out regions. "
          "PDF and SVG are vector; PNG is 600 dpi.", "", "## Files", ""]

    # confusion matrices
    conf = {}
    for name, task in CONFUSION.items():
        cats = list(get_classification_mode(VRHMM_MODE[task]))
        g = preds[preds.task == task].rename(columns={"label": "true_category", "pred": "predicted_category"})
        conf[task] = (g, cats)
        maj = float(summary[(summary.analysis == PRIMARY) & (summary.task == task)].majority_acc.iloc[0])
        fig, cm, acc, bal = plot_confusion_matrix(g, task, cats, maj)
        files = save(fig, figdir, name)
        md.append(f"- `{files[0]}` (+ `.svg`, `.png`): confusion matrix, {task}, counts and row % (n = {len(g)}; accuracy "
                  f"{acc:.1f}%, balanced accuracy {bal:.1f}%, training-fold majority-class accuracy {100 * maj:.1f}%).")

    # pairwise matrix (vrhmm, unchanged): keep the figure that the function closes, then save all formats
    res = {key[r.df_index]: {"amino_acid": r.aa, "all_scores": {aa: getattr(r, f"ll_{aa}") for aa in AAS}}
           for r in scores.itertuples()}
    close = plt.close
    plt.close = lambda *a, **k: None
    try:
        mat = plot_pairwise_classification_matrix(res, metric="accuracy", amino_acids=AAS, save_path=None,
                                                  title="Pairwise Amino Acid Classification Accuracy", save_text_report=False)
        fig = plt.gcf()
    finally:
        plt.close = close
    files = save(fig, figdir, "suppfig10_pairwise_accuracy")
    mat.to_csv(tabdir / "pairwise_accuracy_matrix.csv")
    lower = mat.to_numpy()[np.tril_indices(len(AAS), -1)]
    md.append(f"- `{files[0]}` (+ `.svg`, `.png`): Supp. Fig. 10, pairwise HMM accuracy (share of the two classes' held-out "
              f"regions in which the own-class score beats the other), lower triangle, RdYlGn 0.5-1.0. Mean over the 190 "
              f"pairs {np.nanmean(lower):.3f}; {int((lower < 0.5).sum())} pairs below 0.5 (drawn at the colour-scale floor).")

    # tracebacks (published style)
    tb_md = ["", "## Traceback panels", "",
             "Example rule (fixed before plotting): margin = own-class forward log-likelihood minus the best of the other "
             "models in the comparison; the example is the held-out region of that class whose margin is closest to the "
             "class median (ties: smallest df_index). Titles give each model's forward log-likelihood (the classification "
             "score) to one decimal, as in the published panels; the Viterbi log-probability of the plotted path is also "
             "listed.", "",
             "Style recreated from the published panels (results/04_14_26_ALL_IQR_TRACES_35_SEGS_MODEL_TESTS/plots/"
             "*_plots.pdf): colours, line styles and widths, markers, alphas, font sizes, axes and the 5-column legend "
             "(model entries, one blank slot, the path-event key). Differences: one example region per figure (the "
             "published figures show four regions side by side), so each figure is one panel with the legend below. "
             "Legends list only what is drawn in that panel: skip, slip, self-loop and insert entries (and the skip and "
             "slip arrows) appear only when the panel's paths contain that step, and the published Start and End entries "
             "are left out because no start or end markers are drawn.", "",
             "Each model's line (with its overlays and arrows) is offset horizontally by a fixed amount per model slot, the "
             f"same in every single panel ({', '.join('0' if o == 0 else f'{o:+.3g}' for o in MODEL_OFFSETS)} match states for the 1st to 5th model), "
             f"so overlapping paths stay visible; the composites use {COMPOSITE_OFFSET_SCALE} x these offsets. Every "
             f"traceback legend is titled '{OFFSET_NOTE}'. Plotted x positions are therefore not exact integer state "
             "indices.", ""]
    tb = {}
    for name, (cls, models, fmt) in TRACEBACKS.items():
        best, med, n_cls = pick_example(scores, cls, models)
        k = int(best.df_index)
        pr = paths[paths.df_index == k].set_index("model")
        labels = [fmt.format(c=cls, m=m) for m in models]
        tb[cls] = ([pr.loc[m, "path"].split() for m in models], labels)
        fig = plt.figure(figsize=(7.2, 5.0))
        ax = fig.add_axes([0.08, 0.2, 6.51 / 7.2, 3.61 / 5.0])            # the published panel size, 6.51 x 3.61 in
        plot_traceback(ax, [pr.loc[m, "path"].split() for m in models])
        ax.set_title(f"{key[k]}\nTrue: {cls} | " + " | ".join(f"{lab}: {best[f'll_{m}']:.1f}" for lab, m in zip(labels, models)),
                     fontsize=9, fontweight="bold")
        traceback_legend(fig, labels, (0.08 + 6.51 / 7.2 / 2, 0.075), path_kinds(tb[cls][0]))
        files = save(fig, figdir, name)
        md.append(f"- `{files[0]}` (+ `.svg`, `.png`): Viterbi traceback, {cls} region vs the {'/'.join(models)} models "
                  "(published style: match-state index against segment number, one line per model).")
        tb_md += [f"### `{name}`: {cls} vs {'/'.join(models)}", "",
                  f"Region `{key[k]}` (df_index {k}, held out in {best.split}; 20-way prediction {best.pred_20way}). Margin "
                  f"{best.margin:.2f} vs class median {med:.2f} over {n_cls} held-out {cls} regions.", "",
                  "| Line (model) | Forward log P (title) | Viterbi log P | Path states | Skips | Slips | Inserts |",
                  "|---|---|---|---|---|---|---|"]
        for lab, m in zip(labels, models):
            st = pr.loc[m, "path"].split()
            tb_md.append(f"| {lab} | {best[f'll_{m}']:.2f} | {pr.loc[m, 'viterbi_logp']:.2f} | {len(st)} | "
                         f"{sum(s.startswith('Skip') for s in st)} | {sum(s.startswith('Slip') for s in st)} | "
                         f"{sum(s.startswith('Insert') for s in st)} |")
        tb_md.append("")

    # composites: Figure 6 (A binary charge, B three-way charge, C polarity) and Supp. Fig. 11 (size)
    fig = plt.figure(figsize=FIG6_SIZE)
    for row, (letter, task, cls, legend, loc) in enumerate((("A", "2way", "H", "full", "center right"),
                                                            ("B", "3way", "V", "models", "lower right"),
                                                            ("C", "polarity", "P", "models", "lower right"))):
        top, last = FIG6_ROW_TOPS[row], row == len(FIG6_ROW_TOPS) - 1
        g, cats = conf[task]
        draw_confusion_compact(add_axes_in(fig, FIG6_CM, top), g, cats, ytick_rotation=90 if task == "2way" else 0, xlabel=last,
                               label_x=(TRUE_LABEL_X - FIG6_CM[0]) / FIG6_CM[2])
        draw_traceback_compact(add_axes_in(fig, FIG6_TB, top), *tb[cls], legend=legend, loc=loc, xlabel=last)
        fig.text(0.02 / FIG6_SIZE[0], 1 - (top + 0.04) / FIG6_SIZE[1], letter, fontsize=12, fontweight="bold", va="top")
    files = save(fig, figdir, "fig6_composite", tight=False)
    md.append(f"- `{files[0]}` (+ `.svg`, `.png`): Figure 6 assembled, {FIG6_SIZE[0]} x {FIG6_SIZE[1]} in: rows A binary "
              "charge, B three-way charge, C polarity; confusion matrix left, traceback right (H vs H/D, V vs V/H/D, P vs "
              "P/H/D/T).")
    fig = plt.figure(figsize=S11_SIZE)
    g, cats = conf["5way_size"]
    draw_confusion_compact(add_axes_in(fig, S11_CM), g, cats, xlabel=False, fs=6, label_x=(TRUE_LABEL_X - S11_CM[0]) / S11_CM[2])
    draw_traceback_compact(add_axes_in(fig, S11_TB), *tb["W"], legend="ideal", loc="center right", xlabel=False)
    files = save(fig, figdir, "suppfig11_composite", tight=False)
    md.append(f"- `{files[0]}` (+ `.svg`, `.png`): Supp. Fig. 11 assembled, {S11_SIZE[0]} x {S11_SIZE[1]} in: size "
              "confusion matrix left, W vs W/E/H/S/D traceback right.")

    # Fig. 7 and Supp. Fig. 12: state-coloured signal under the region's own amino-acid model
    wanted = {k for panels in STATE_FIGS.values() for _, _, k in panels if k is not None}
    choice = {}
    for name, panels in STATE_FIGS.items():
        for letter, cls, k in panels:
            if k is None:
                k, med, n_cls, n_el = pick_slip_example(scores, paths, cls)
                choice[(name, letter)] = (k, f"fallback rule: held-out {cls} region with \u2265 1 slip call under the {cls} "
                                             f"model ({n_el} of {n_cls}) whose margin is closest to the class median "
                                             f"({med:.2f})")
            else:
                choice[(name, letter)] = (k, "published region, kept (held out; event still shown)")
            wanted.add(k)
    segs = {int(r["df_index"]): [np.asarray(x, float) for x in r["cleaned_segments"]]
            for r in pickle.load(open(args.segments, "rb")) if int(r["df_index"]) in wanted}
    st_md = ["", "## Fig. 7 and Supp. Fig. 12 (state-coloured signal)", "",
             "Each panel shows one held-out region's 35 Dynp segments (concatenated, z-normalised over the region), coloured "
             "by the state that emitted each segment on the Viterbi path under the region's own amino-acid model (the model "
             "of the fold that held it out). The number above each segment is its match state; light-green segments are "
             "inserted (interpolated) states. Red dashed line: segment entered by a skip. Blue dashed line and blue band: "
             "segment entered by a backslip (slip). Green dotted line: start of an inserted segment. The published panels "
             "used the best-scoring model's path (G 955 under T, P 6009 under Y; D 3547 under D).", "",
             "Region rule: the published region if held out and the event is still shown; otherwise the fallback rule "
             "(held-out regions of the same amino acid with at least one slip call under their own model; own-class "
             "margin, ll_own minus the best of the other 19 models, closest to the class median; ties: smallest "
             "df_index). Fig. 7B's published region (D 3547, held out in fold2) is replaced: the published Fig. 7B "
             "caption does not describe a skip, and under the D model its path has no slip call.", ""]
    for name, panels in STATE_FIGS.items():
        (W, H), boxes = STATE_SIZES[name]
        fig = plt.figure(figsize=(W, H))
        st_md += [f"### `{name}`", ""]
        for (letter, cls, _), box in zip(panels, boxes):
            k, why = choice[(name, letter)]
            r = scores[scores.df_index == k].iloc[0]
            pr = paths[(paths.df_index == k) & (paths.model == cls)].iloc[0]
            ax = add_axes_in(fig, box)
            events = plot_state_signal(ax, segs[k], pr.path.split(), legend=letter == "A")
            if letter:
                fig.text(0.02 / W, 1 - (box[1] - 0.02) / H, letter, fontsize=10, fontweight="bold", va="top")
            margin = r[f"ll_{cls}"] - max(r[f"ll_{a}"] for a in AAS if a != cls)
            st_md += [f"**{'Panel ' + letter if letter else 'Panel'}: {cls}**, region `{key[k]}` (df_index {k}; "
                      f"{reg.loc[k, 'construct']}; {sum(len(x) for x in segs[k])} samples; held out in {pr.split}; "
                      f"20-way prediction {r.pred_20way}). {why}.", "",
                      f"- {cls} model: forward log P {pr.forward_logp:.2f}, Viterbi log P of the plotted path "
                      f"{pr.viterbi_logp:.2f}; own-class margin {margin:.2f}.",
                      f"- Path: {sum('slip' in e[4] for e in events)} backslips, "
                      f"{sum('skip' in e[4] for e in events)} skips, {sum('insert' in e[4] for e in events)} inserted "
                      f"segments; {len({int(s.split('_')[1]) for s in pr.path.split() if s.startswith('Match_')})} "
                      "unique match states.",
                      "", "| Segment (0-based) | Start sample | Transition |", "|---|---|---|"]
            st_md += [f"| {j} | {x0} | {describe_event(prev, st, ev)} |" for j, x0, prev, st, ev in events]
            st_md += ["", "Viterbi path (Match_k = k, Insert_i_j = i-j, silent Skip/Slip states omitted): "
                      + " ".join(state_label(s) for s in pr.path.split() if s.startswith(("Match_", "Insert_"))), ""]
        files = save(fig, figdir, name, tight=False)
        md.append(f"- `{files[0]}` (+ `.svg`, `.png`): "
                  + ("Figure 7, state-coloured signal: A G region, B D region" if name.startswith("fig7")
                     else "Supp. Fig. 12, state-coloured signal of a P region")
                  + f", {W} x {H} in (decoded under the region's own model).")

    # path statistics (notebook logic) on each region's path under its best-scoring model
    best_paths = paths.merge(scores[["df_index", "pred_20way"]], left_on=["df_index", "model"], right_on=["df_index", "pred_20way"])
    res = {key[r.df_index]: {"amino_acid": r.aa, "full_path": r.path.split()} for r in best_paths.itertuples()}
    for mode in ("2way", "3way", "polarity", "5way_size"):
        t = build_path_stats_table(res, mode, tabdir / f"path_stats_{mode}.csv")
        md.append(f"- `tables/path_stats_{mode}.csv`: HMM path statistics per {mode} class ({int(t.n_traces.sum())} regions): "
                  "regions, then mean and SD of unique matched states, backslips, skips, self-loops and inserts on the "
                  "Viterbi path of each region's best-scoring model (notebook `build_path_stats_table`).")
    md.append("- `tables/pairwise_accuracy_matrix.csv`: the Supp. Fig. 10 matrix.")
    md.append("- Supplement table numbers: Table S8 = `path_stats_2way.csv`, Table S9 = `path_stats_3way.csv`, Table S10 = "
              "`path_stats_polarity.csv`, Table S11 = `path_stats_5way_size.csv`.")
    md += ["", "## Confusion-matrix style", "",
           "Adapted from vrhmm's `plot_confusion_matrix` to match the published panels; the changes are listed in the "
           "docstring of `hmm_figures.plot_confusion_matrix`. Two deliberate departures from the published look: a second "
           "title line with balanced accuracy and the majority-class accuracy, and per-row recall labels that no longer "
           "overlap the colour bar."]
    md += ["", "## Composite figures", "",
           "Layout, panel proportions, label style and overall size measured from the revision PDFs (Figure 6 on manuscript "
           "page 20; Supp. Fig. 11 on supplement page 12, both rendered at 8.5 in page width), to about 0.1 in:",
           f"- Figure 6: {FIG6_SIZE[0]} x {FIG6_SIZE[1]} in; three rows {FIG6_ROW_TOPS[1]} in apart; confusion matrix "
           f"{FIG6_CM[2]} x {FIG6_CM[3]} in, traceback {FIG6_TB[2]} x {FIG6_TB[3]} in; bold A/B/C labels at each row's "
           "top-left; 'True' on every row, 'Predicted' and 'Match State Index' on the bottom row only; row A's class labels "
           "rotated as published.",
           f"- Supp. Fig. 11: {S11_SIZE[0]} x {S11_SIZE[1]} in, no panel letters; confusion matrix {S11_CM[2]} x "
           f"{S11_CM[3]} in, traceback {S11_TB[2]} x {S11_TB[3]} in.",
           "- As in the published composites: confusion matrices without colour bar or title, with 'count (row %)' in every "
           "cell including zeros; traceback legends inside the axes (Figure 6 A: models plus the path-event key for the step types drawn; B and C: "
           "models; Supp. Fig. 11: models plus the ideal alignment); no traceback titles. Traceback lines and markers are "
           f"drawn at {COMPOSITE_SCALE} x the single-panel widths, and the per-model offsets are {COMPOSITE_OFFSET_SCALE} x "
           f"the single-panel offsets ({', '.join('0' if o == 0 else f'{o * COMPOSITE_OFFSET_SCALE:+.3g}' for o in MODEL_OFFSETS)} "
           "match states for the 1st to 5th model; same legend title).",
           "- Data, example regions and log-probabilities are the same as the single panels (listed below)."]
    md += tb_md + st_md
    (out / "FIGURES.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    sys.exit(main())
