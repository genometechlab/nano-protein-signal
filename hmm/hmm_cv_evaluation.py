"""Cross-validated evaluation of the Dynp-34 profile HMM (the analysis reported in the paper and response letter).

Data
  Regions: the records of the Dynp-34 segment pickle (--segments), one per PASTOR variable region, each cut into 35
  segments of the cleaned signal. Label = variable_region; molecule = (run, int(channel)); construct = the metadata
  prefix (e.g. "HDKER" in "HDKER_1"); length = number of samples.
  In-bounds: length within the 25th-75th percentiles of all region lengths (np.percentile, linear; [2323, 3194.5] for
  the 2,987 regions). The bounds use no labels and no folds.
  Folds: test_fold from --folds (df_index, trace_id, test_fold), the 5 folds shared with the LSTM evaluation.

HMM, per held-out fold, built from the in-bounds regions of the four training folds only
  Profiles: per amino acid, the tslearn DBA barycenter of the per-region z-scored segments (dtw.barycenter.
    build_centroids) and 'segment'-mode variances from the first 10 training regions in df_index order
    (vrhmm SegmentVarianceCollector). No hand-set variance boosts.
  Variance scale: one global scale from SCALES, chosen by 20-way balanced accuracy on a grouped inner split of the
    training regions (StratifiedGroupKFold(5, shuffle=True, random_state=20261015), groups = molecule, stratified by
    amino acid, split 0 = inner validation). Ties go to the smaller scale.
  Transitions, primary: one-pass Viterbi re-estimation. Build the models with the manuscript transitions (vrhmm default
    config), decode every training region with its own class model, count the transition types, apply add-one
    smoothing and rebuild. The same procedure runs inside the inner split for every candidate scale.
  Transitions, sensitivity: the manuscript transitions, with the scale tuned separately on the same inner split.
  Scoring: every held-out in-bounds region is scored by all 20 class models (forward log-likelihood,
    model.log_probability, as in vrhmm's HMMClassifier). The 20-way prediction is the best-scoring model. Grouped tasks
    take the best score within each vrhmm category; 2-way charge uses only K/R/H and D/E regions.
  Observations: z-normalised segment means (vrhmm SignalProcessor._z_normalize_means).

Other analyses
  Cross-context: the construct split {AVLIM, FYWCP, GNQST, HDKER} vs {KSILC, PRMQE, TWAFH, VGDNY} (each amino acid
    once per side). Train on one side's in-bounds regions and test on the other side's, in both directions, with the
    same pipeline.
  Summary-statistics baseline: logistic regression (StandardScaler, lbfgs, max_iter 5000) on region mean, SD and length,
    one model per task, trained on all training-fold regions (cross-context: all training-side regions) and predicting
    every held-out region.
  Drop-66 sensitivity: the regions of molecules whose traces sit in different folds are removed before scoring.

Metrics
  Pooled out-of-fold accuracy, balanced accuracy and macro-F1, with 95% CIs from 1,000 bootstrap resamples of
  molecules (seed 20261016); per-fold mean +- SD of balanced accuracy; training-fold majority-class baseline. Paired
  differences use the same resamples on identical regions (bootstrap p = 2 x the smaller tail share).

Outputs (--out)
  regions.csv          one row per region: df_index, aa, construct, run, channel, molecule, length, in_bounds,
                       test_fold, side (cross-context).
  scores.csv.gz        the 20 per-class log-likelihoods (ll_A ... ll_Y) per held-out region, per analysis and
                       transition set, with the chosen variance scale and the 20-way prediction.
  predictions.csv.gz   per-region predictions for every task and analysis (label, pred).
  viterbi_paths.csv.gz the primary's Viterbi path of every held-out in-bounds region under each of the 20 fold models
                       (start and end states removed, as in vrhmm's full_path), with forward and Viterbi log-probability.
  profiles.csv         the primary fold models' emission mean and SD per (split, amino acid, state).
  transitions.csv, tuning.csv, length_bounds.csv, summary.csv, paired.csv.

Usage (from the repository root):
  python hmm/hmm_cv_evaluation.py --folds hmm/cv_folds.csv \
      --segments hmm/vrhmm/data/denoise_segment_dynp_parallel_34_BkP_full_pastor_norm_features_clean_1751_4570_min_size_30.pkl \
      --out results/hmm_cv --n-jobs 8
"""
import argparse
import copy
import os
import pickle
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "hmm"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dtw.barycenter import build_centroids  # noqa: E402
from dtw.preprocessing import zscore_segments  # noqa: E402
from vrhmm.config import CONFIG  # noqa: E402
from vrhmm.core import HMMConstructor  # noqa: E402
from vrhmm.processing.signal_processor import SignalProcessor  # noqa: E402
from vrhmm.segmentation import SegmentVarianceCollector  # noqa: E402
from vrhmm.utils.amino_acids import get_classification_mode  # noqa: E402

AAS = list("ACDEFGHIKLMNPQRSTVWY")
SCALES = [0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]
INNER_SEED = 20261015
BOOT_SEED = 20261016
N_BOOT = 1000
N_STATES = 35
CONTEXT_A = {"AVLIM", "FYWCP", "GNQST", "HDKER"}
TASKS = {"20way": None, "2way": "2way", "3way": "3way", "5way_size": "5way_size", "polarity": "biological"}
T0 = copy.deepcopy(CONFIG["hmm"]["transitions"])
TRANSITION_KEYS = [["match_self_loop", "forward", "to_skip", "to_slip", "to_end", "to_insert"],
                   ["insert_self_loop", "insert_to_match"], ["skip_continue", "skip_to_match"],
                   ["slip_continue", "slip_to_match"]]
D = None   # region table indexed by df_index, shared with the worker processes


# ── Data ──────────────────────────────────────────────────────────────

def load_regions(folds_path, segments_path):
    recs = pickle.load(open(segments_path, "rb"))
    rows = []
    for r in recs:
        segs = [np.asarray(s, float) for s in r["cleaned_segments"]]
        rows.append(dict(df_index=int(r["df_index"]), aa=r["variable_region"], run=r["run"], channel=float(r["channel"]),
                         construct=str(r["metadata"]).split("_")[0], length=int(sum(len(s) for s in segs)), segs=segs))
    d = pd.DataFrame(rows)
    folds = pd.read_csv(folds_path)[["df_index", "test_fold"]]
    d = d.merge(folds, on="df_index", how="inner", validate="one_to_one")
    if len(d) != len(recs) or len(d) != len(folds):
        raise ValueError(f"fold file and segment pickle disagree: {len(folds)} folds, {len(recs)} records, {len(d)} matched")
    d["molecule"] = d.run + "_" + d.channel.astype(int).astype(str)
    lo, hi = np.percentile(d.length, [25, 75])
    d["in_bounds"] = d.length.between(lo, hi)
    d["side"] = np.where(d.construct.isin(CONTEXT_A), "A", "B")
    return d.sort_values("df_index").reset_index(drop=True), (float(lo), float(hi))


def zmeans(segs):
    return SignalProcessor._z_normalize_means(np.array([np.mean(s) for s in segs], float))


# ── HMM ───────────────────────────────────────────────────────────────

def build_profiles(fit):
    """DBA barycenter and first-10 segment variances per amino acid, in df_index order."""
    prof = {}
    for aa, g in fit.groupby("aa"):
        bary = build_centroids({aa: [zscore_segments(s) for s in g.segs]}, [aa])[aa]
        coll = SegmentVarianceCollector()
        for segs in list(g.segs)[:10]:
            coll.add_signal_variances([float(np.var(s)) for s in segs])
        prof[aa] = ([np.asarray(b) for b in bary], coll.get_average_variances())
    return prof


def build_models(prof, scale, transitions):
    """Returns ({aa: model}, {aa: {state: (mean, sd)}})."""
    cfg = copy.deepcopy(CONFIG["hmm"])
    cfg["transitions"] = transitions
    models, stats = {}, {}
    for aa, (b, v) in prof.items():
        models[aa], stats[aa] = HMMConstructor(config=cfg, variance_mode="segment", variance_scale=scale).build_hmm_from_arrays(aa, b, v)
    return models, stats


def viterbi_names(model, obs):
    lp, path = model.viterbi(np.asarray(obs, float))
    if path is None or not np.isfinite(lp):
        return None, lp
    return [s.name for _, s in path], lp


def count_transitions(names, n_states, cnt):
    for a, b in zip(names[:-1], names[1:]):
        if a.startswith("Match_"):
            i = int(a.split("_")[1])
            if b == a:
                cnt["match_self_loop"] += 1
            elif b == f"Match_{i + 1}":
                cnt["forward"] += 1
            elif b.startswith("Insert_"):
                cnt["to_insert"] += 1
            elif b.startswith("Skip_"):
                cnt["to_skip"] += 1
            elif b.startswith("Slip_"):
                cnt["to_slip"] += 1
            elif b.endswith("-end") and i < n_states - 1:
                cnt["to_end"] += 1
        elif a.startswith("Insert_"):
            cnt["insert_self_loop" if b == a else "insert_to_match"] += 1
        elif a.startswith("Skip_"):
            cnt["skip_continue" if b.startswith("Skip_") else "skip_to_match"] += 1
        elif a.startswith("Slip_"):
            cnt["slip_continue" if b.startswith("Slip_") else "slip_to_match"] += 1


def reestimated_probabilities(cnt):
    t = {}
    for group in TRANSITION_KEYS:
        tot = sum(cnt[k] + 1 for k in group)          # add-one smoothing
        for k in group:
            t[k] = (cnt[k] + 1) / tot
    return t


def reestimate(models, segs, aas):
    cnt = Counter()
    for s, aa in zip(segs, aas):
        names, _ = viterbi_names(models[aa], zmeans(s))
        if names:
            count_transitions(names, N_STATES, cnt)
    T = dict(T0)
    T.update(reestimated_probabilities(cnt))
    return T, cnt


def score_matrix(models, segs):
    out = np.full((len(segs), len(AAS)), -np.inf)
    for i, s in enumerate(segs):
        z = zmeans(s)
        for j, aa in enumerate(AAS):
            try:
                out[i, j] = models[aa].log_probability(z)
            except Exception:  # noqa: BLE001  (an undecodable region keeps -inf, as in HMMClassifier)
                pass
    return out


def fit(tr, transitions):
    """Inner-split scale choice, then the final models on all of tr."""
    tr = tr.sort_values("df_index")
    itr, iva = next(StratifiedGroupKFold(5, shuffle=True, random_state=INNER_SEED).split(tr, tr.aa, tr.molecule.to_numpy()))
    a, b = tr.iloc[itr], tr.iloc[iva]
    prof_in = build_profiles(a)
    tune = []
    for s in SCALES:
        m, _ = build_models(prof_in, s, T0)
        if transitions == "reestimated":
            m, _ = build_models(prof_in, s, reestimate(m, a.segs, a.aa)[0])
        p = [AAS[k] for k in score_matrix(m, list(b.segs)).argmax(1)]
        tune.append(dict(scale=s, inner_bal_acc=balanced_accuracy_score(b.aa, p), n_inner_train=len(a), n_inner_val=len(b)))
    best = max(tune, key=lambda r: (r["inner_bal_acc"], -r["scale"]))["scale"]
    prof = build_profiles(tr)
    models, stats = build_models(prof, best, T0)
    T, cnt = T0, None
    if transitions == "reestimated":
        T, cnt = reestimate(models, tr.segs, tr.aa)
        models, stats = build_models(prof, best, T)
    return models, stats, best, tune, T, cnt


def run_split(task):
    analysis, split, train_ids, test_ids = task
    tr, te = D.loc[train_ids], D.loc[test_ids]
    scores, tunes, trans, paths, profiles = [], [], [], [], []
    for tset in ("reestimated", "manuscript"):
        models, stats, scale, tune, T, cnt = fit(tr, tset)
        sc = score_matrix(models, list(te.segs))
        for (_, r), v in zip(te.iterrows(), sc):
            scores.append(dict(analysis=analysis, split=split, transitions=tset, scale=scale, df_index=r.df_index, aa=r.aa,
                               pred_20way=AAS[int(np.argmax(v))], decoded=bool(np.isfinite(v).any()),
                               **{f"ll_{aa}": x for aa, x in zip(AAS, v)}))
        tunes += [dict(analysis=analysis, split=split, transitions=tset, **t, chosen=t["scale"] == scale) for t in tune]
        keys = [k for g in TRANSITION_KEYS for k in g]
        trans.append(dict(analysis=analysis, split=split, transitions=tset, scale=scale, n_train=len(tr),
                          **{k: T.get(k) for k in keys}, **({f"count_{k}": cnt[k] for k in keys} if cnt is not None else {})))
        if analysis == "primary" and tset == "reestimated":
            for (_, r), v in zip(te.iterrows(), sc):
                z = zmeans(r.segs)
                for j, aa in enumerate(AAS):
                    names, vlp = viterbi_names(models[aa], z)
                    full = [n for n in (names or []) if n and "start" not in n.lower() and "end" not in n.lower()]
                    paths.append(dict(split=split, df_index=r.df_index, aa=r.aa, model=aa, forward_logp=v[j],
                                      viterbi_logp=float(vlp), path=" ".join(full)))
            profiles += [dict(split=split, amino_acid=aa, state=int(k), mean=mu, std=sd)
                         for aa, st in stats.items() for k, (mu, sd) in st.items()]
    print(f"{analysis} {split}: n_train {len(tr)}, n_test {len(te)} done", flush=True)
    return scores, tunes, trans, paths, profiles


def run_baseline(d, cross_context=False):
    """Logistic regression on region mean, SD and length; one model per task and fold (or cross-context side)."""
    sig = [np.concatenate(s) for s in d.segs]
    x = np.c_[[np.mean(s) for s in sig], [np.std(s) for s in sig], d.length]
    if cross_context:
        a = (d.side == "A").to_numpy()
        splits = [("train_A_test_B", a, ~a), ("train_B_test_A", ~a, a)]
    else:
        splits = [(f"fold{f}", (d.test_fold != f).to_numpy(), (d.test_fold == f).to_numpy()) for f in sorted(d.test_fold.unique())]
    rows = []
    for task in TASKS:
        y = d.aa.map(category_map(task))
        keep = y.notna().to_numpy()
        for split, trm, tem in splits:
            trm, tem = keep & trm, keep & tem
            lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(x[trm], y[trm])
            rows += [dict(split=split, df_index=k, task=task, label=t, pred=p)
                     for k, t, p in zip(d.df_index[tem], y[tem], lr.predict(x[tem]))]
    return pd.DataFrame(rows)


# ── Tasks and metrics ─────────────────────────────────────────────────

def category_map(task):
    mode = TASKS[task]
    return {aa: aa for aa in AAS} if mode is None else {aa: c for c, aas in get_classification_mode(mode).items() for aa in aas}


def task_predictions(sc):
    """Per-region predictions for every task from the 20 per-class scores (best score within each category)."""
    out = []
    for task in TASKS:
        cm = category_map(task)
        y = sc.aa.map(cm)
        keep = y.notna().to_numpy()
        g = sc[keep]
        cats = sorted(set(cm.values()))
        best = np.stack([g[[f"ll_{aa}" for aa in AAS if cm.get(aa) == c]].max(axis=1).to_numpy() for c in cats], 1)
        out.append(pd.DataFrame({"analysis": g.analysis.to_numpy(), "transitions": g.transitions.to_numpy(),
                                 "split": g.split.to_numpy(), "df_index": g.df_index.to_numpy(), "task": task,
                                 "label": y[keep].to_numpy(), "pred": np.array(cats)[best.argmax(1)]}))
    return pd.concat(out, ignore_index=True)


class Boot:
    """Molecule bootstrap: the same resamples for every metric and comparison."""

    def __init__(self, regions):
        self.mol = regions.set_index("df_index").molecule
        rng = np.random.default_rng(BOOT_SEED)
        um = np.array(sorted(self.mol.unique()))
        self.draws = [rng.choice(um, len(um)) for _ in range(N_BOOT)]

    def idx(self, df_index):
        m = self.mol.loc[df_index].to_numpy()
        pos = {}
        for i, x in enumerate(m):
            pos.setdefault(x, []).append(i)
        for d in self.draws:
            yield np.concatenate([pos[x] for x in d if x in pos] or [np.array([], int)]).astype(int)


def encode(y, p):
    labs = sorted(set(y) | set(p))
    ix = {c: i for i, c in enumerate(labs)}
    return np.array([ix[c] for c in y]), np.array([ix[c] for c in p]), len(labs)


def metric(y, p, kind, K):
    """Accuracy, balanced accuracy or macro-F1 (classes present in y), as sklearn defines them."""
    if len(y) == 0:
        return np.nan
    cm = np.bincount(y * K + p, minlength=K * K).reshape(K, K)
    d, rs, cs = np.diag(cm), cm.sum(1), cm.sum(0)
    pres = rs > 0
    if kind == "acc":
        return d.sum() / len(y)
    rec = np.where(pres, d / np.maximum(rs, 1), 0.0)
    if kind == "bal_acc":
        return rec[pres].mean()
    prec = np.where(cs > 0, d / np.maximum(cs, 1), 0.0)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    return f1[pres].mean()


def summarize(pr, groups, boot, label, majority=None):
    rows = []
    for task, g in pr.groupby("task", sort=False):
        g = g.reset_index(drop=True)
        y, p, K = encode(g.label.to_numpy(), g.pred.to_numpy())
        r = dict(analysis=label, task=task, n=len(g), n_classes=g.label.nunique())
        for k in ("acc", "bal_acc", "macro_f1"):
            r[k] = metric(y, p, k, K)
        bs = {k: [] for k in ("acc", "bal_acc", "macro_f1")}
        for ii in boot.idx(g.df_index):
            for k in bs:
                bs[k].append(metric(y[ii], p[ii], k, K))
        for k in bs:
            r[f"{k}_ci_lo"], r[f"{k}_ci_hi"] = np.percentile(bs[k], [2.5, 97.5])
        f = groups.loc[g.df_index].to_numpy()
        pf = [metric(y[f == x], p[f == x], "bal_acc", K) for x in sorted(set(f))]
        r["bal_acc_fold_mean"], r["bal_acc_fold_sd"] = float(np.mean(pf)), float(np.std(pf, ddof=1))
        if majority is not None:
            maj = np.array([majority[(task, x)] for x in f])
            r["majority_acc"] = accuracy_score(g.label, maj)
            r["majority_bal_acc"] = balanced_accuracy_score(g.label, maj)
        rows.append(r)
    return rows


def paired(a, b, boot, label_a, label_b, task, kinds=("bal_acc", "acc")):
    """a - b on identical regions of one task."""
    m = a[a.task == task].merge(b[b.task == task], on=["df_index", "label"], suffixes=("_a", "_b"))
    labs = sorted(set(m.label) | set(m.pred_a) | set(m.pred_b))
    ix = {c: i for i, c in enumerate(labs)}
    K = len(labs)
    y, pa, pb = (np.array([ix[c] for c in m[col]]) for col in ("label", "pred_a", "pred_b"))
    rows = []
    for k in kinds:
        ds = np.array([metric(y[ii], pa[ii], k, K) - metric(y[ii], pb[ii], k, K) for ii in boot.idx(m.df_index)])
        rows.append(dict(task=task, a=label_a, b=label_b, metric=k, n=len(m), a_value=metric(y, pa, k, K),
                         b_value=metric(y, pb, k, K), diff=metric(y, pa, k, K) - metric(y, pb, k, K),
                         ci_lo=np.percentile(ds, 2.5), ci_hi=np.percentile(ds, 97.5),
                         boot_p=min(1.0, 2 * min((ds <= 0).mean(), (ds >= 0).mean()))))
    return rows


def majority_classes(d, eval_ids):
    """Training-fold majority class per (task, fold) within the evaluation set."""
    e = d[d.df_index.isin(eval_ids)]
    out = {}
    for task in TASKS:
        y = e.aa.map(category_map(task))
        for f in sorted(e.test_fold.unique()):
            out[(task, f)] = y[(e.test_fold != f).to_numpy() & y.notna().to_numpy()].value_counts().idxmax()
    return out


def report(d, preds, out):
    folds = d.set_index("df_index").test_fold
    sides = d.set_index("df_index").side
    boot = Boot(d)
    ib, all_ids = set(d.df_index[d.in_bounds]), set(d.df_index)
    drop66 = set(d.df_index[d.groupby("molecule").test_fold.transform("nunique") > 1])
    print(f"molecules spanning folds: {d[d.df_index.isin(drop66)].molecule.nunique()} ({len(drop66)} regions)", flush=True)
    maj_ib, maj_all = majority_classes(d, ib), majority_classes(d, all_ids)

    def sub(pr, ids):
        return pr[pr.df_index.isin(ids)]

    summ, pairs = [], []
    for drop, tag in ((set(), ""), (drop66, " [drop-66]")):
        k_ib, k_all = ib - drop, all_ids - drop
        for key, lab in (("primary", "HMM primary (Dynp-34, re-estimated transitions)"),
                         ("sensitivity", "HMM, manuscript transitions"), ("lr_baseline", "summary-statistics LR")):
            summ += summarize(sub(preds[key], k_ib), folds, boot, f"{lab} | in-bounds{tag}", maj_ib)
        summ += summarize(sub(preds["lr_baseline"], k_all), folds, boot, f"summary-statistics LR | all regions{tag}", maj_all)
        summ += summarize(sub(preds["cross_context"], k_ib), sides, boot, f"cross-context HMM (re-estimated) | in-bounds{tag}")
        summ += summarize(sub(preds["cross_context_manuscript"], k_ib), sides, boot,
                          f"cross-context HMM, manuscript transitions | in-bounds{tag}")
        summ += summarize(sub(preds["lr_baseline_cross_context"], k_ib), sides, boot, f"cross-context LR | in-bounds{tag}")
        for task in TASKS:
            kinds = ("bal_acc", "acc") if task == "20way" else ("bal_acc",)
            pairs += [dict(r, comparison=f"in-bounds{tag}") for r in
                      paired(sub(preds["primary"], k_ib), sub(preds["lr_baseline"], k_ib), boot, "HMM primary", "LR", task, kinds)]
            pairs += [dict(r, comparison=f"in-bounds{tag}") for r in
                      paired(sub(preds["sensitivity"], k_ib), sub(preds["primary"], k_ib), boot, "manuscript transitions",
                             "HMM primary", task, ("bal_acc",))]
        pairs += [dict(r, comparison=f"in-bounds{tag}") for r in
                  paired(sub(preds["cross_context"], k_ib), sub(preds["primary"], k_ib), boot, "cross-context", "HMM primary",
                         "20way", ("bal_acc",))]
        for task in TASKS:
            pairs += [dict(r, comparison=f"cross-context in-bounds{tag}") for r in
                      paired(sub(preds["cross_context"], k_ib), sub(preds["lr_baseline_cross_context"], k_ib), boot,
                             "HMM cross-context", "LR cross-context", task, ("bal_acc",))]
    S, P = pd.DataFrame(summ), pd.DataFrame(pairs)
    S.to_csv(out / "summary.csv", index=False)
    P.to_csv(out / "paired.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    print(S[["analysis", "task", "n", "acc", "bal_acc", "bal_acc_ci_lo", "bal_acc_ci_hi", "macro_f1", "bal_acc_fold_mean",
             "bal_acc_fold_sd"]].round(3).to_string(index=False))
    print(P[["comparison", "task", "a", "b", "metric", "n", "diff", "ci_lo", "ci_hi", "boot_p"]].round(3).to_string(index=False))


def main():
    global D
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--folds", default="hmm/cv_folds.csv", help="CSV with df_index, trace_id, test_fold")
    ap.add_argument("--segments", default="hmm/vrhmm/data/denoise_segment_dynp_parallel_34_BkP_full_pastor_norm_features_"
                                          "clean_1751_4570_min_size_30.pkl", help="Dynp-34 segment pickle")
    ap.add_argument("--out", default="results/hmm_cv", help="output directory")
    ap.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    d, (lo, hi) = load_regions(args.folds, args.segments)
    if not (d.segs.map(len) == N_STATES).all():
        raise ValueError("every region must have 35 segments (Dynp with n_bkps = 34)")
    print(f"{len(d)} regions, {d.molecule.nunique()} molecules; in-bounds [{lo}, {hi}]: {int(d.in_bounds.sum())}", flush=True)
    pd.DataFrame([dict(q25=lo, q75=hi, n_regions=len(d), n_in_bounds=int(d.in_bounds.sum()))]).to_csv(out / "length_bounds.csv", index=False)
    d.drop(columns="segs").to_csv(out / "regions.csv", index=False)

    D = d.set_index("df_index", drop=False).rename_axis(None)
    ib = d[d.in_bounds]
    tasks = [("primary", f"fold{f}", list(ib.df_index[ib.test_fold != f]), list(ib.df_index[ib.test_fold == f]))
             for f in sorted(d.test_fold.unique())]
    tasks += [("cross_context", "train_A_test_B", list(ib.df_index[ib.side == "A"]), list(ib.df_index[ib.side == "B"])),
              ("cross_context", "train_B_test_A", list(ib.df_index[ib.side == "B"]), list(ib.df_index[ib.side == "A"]))]
    with Pool(max(1, min(args.n_jobs, len(tasks)))) as pool:
        res = pool.map(run_split, tasks)
    scores = pd.DataFrame([r for x in res for r in x[0]])
    scores.to_csv(out / "scores.csv.gz", index=False)
    pd.DataFrame([r for x in res for r in x[1]]).to_csv(out / "tuning.csv", index=False)
    pd.DataFrame([r for x in res for r in x[2]]).to_csv(out / "transitions.csv", index=False)
    pd.DataFrame([r for x in res for r in x[3]]).to_csv(out / "viterbi_paths.csv.gz", index=False)
    pd.DataFrame([r for x in res for r in x[4]]).to_csv(out / "profiles.csv", index=False)

    hmm = task_predictions(scores)
    preds = {"primary": hmm[(hmm.analysis == "primary") & (hmm.transitions == "reestimated")],
             "sensitivity": hmm[(hmm.analysis == "primary") & (hmm.transitions == "manuscript")],
             "cross_context": hmm[(hmm.analysis == "cross_context") & (hmm.transitions == "reestimated")],
             "cross_context_manuscript": hmm[(hmm.analysis == "cross_context") & (hmm.transitions == "manuscript")],
             "lr_baseline": run_baseline(d), "lr_baseline_cross_context": run_baseline(d, cross_context=True)}
    names = {"primary": "primary", "sensitivity": "sensitivity_manuscript_transitions", "cross_context": "cross_context",
             "cross_context_manuscript": "cross_context_manuscript_transitions", "lr_baseline": "lr_baseline",
             "lr_baseline_cross_context": "lr_baseline_cross_context"}
    pd.concat([p.assign(analysis=names[k])[["analysis", "split", "df_index", "task", "label", "pred"]] for k, p in preds.items()],
              ignore_index=True).to_csv(out / "predictions.csv.gz", index=False)
    report(d, preds, out)


if __name__ == "__main__":
    sys.exit(main())
