# vrhmm

Variable Region Hidden Markov Model for amino acid classification from nanopore sequencing signals.

vrhmm builds profile HMMs from barycenter templates of nanopore current signals, then classifies new traces by scoring them against each model.

## How it works

1. **Profile construction** — For each amino acid, a barycenter (average shape) of known signals is segmented into discrete levels. Each level becomes a match state with a Gaussian emission distribution parameterized by the segment's mean and variance.

2. **HMM topology** — Each profile HMM contains match, insert, skip, and slip states connected by configurable transition probabilities. Insert states model extra dwell time between expected levels. Skip states allow the model to jump forward when levels are missed. Slip states handle backward translocation (backslips).

3. **Classification** — A new signal is segmented, z-normalized, and scored against all amino acid models. The model with the highest log-likelihood (optionally weighted by path coverage) determines the prediction.

4. **Visualization** — The pipeline generates per-trace HMM state alignments, segment pileups, pairwise accuracy matrices, and backslip/skip distributions.

## Installation

```bash
pip install .
```

With optional dependencies for DTW averaging and network visualization:

```bash
pip install ".[dtw,network]"
```

Requires Python 3.10+.

## Quick start

Classify signals against barycenter profiles:

```bash
vrhmm \
  --signal-file signals.csv \
  --barycenter-file barycenters.json \
  --classification-mode 20way \
  --output-dir results/
```

Use pre-computed profile statistics instead of barycenters:

```bash
vrhmm \
  --signal-file signals.csv \
  --profile-file profiles.csv \
  --classification-mode 20way \
  --output-dir results/
```

Test a specific amino acid against a specific model:

```bash
vrhmm \
  --signal-file signals.csv \
  --barycenter-file barycenters.json \
  --test-aa K \
  --model-aa K \
  --output-dir results/
```

Cross-validate (test one AA's signals against a different AA's model):

```bash
vrhmm \
  --signal-file signals.csv \
  --barycenter-file barycenters.json \
  --test-aa K \
  --model-aa R \
  --output-dir results/
```

## Classification modes

| Mode | Categories | Description |
|------|-----------|-------------|
| `20way` | 20 | Individual amino acid identification |
| `4way` | 4 | Positive, negative, big, small |
| `3way` | 3 | Positive, negative, neutral |
| `2way` | 2 | Positive vs negative charge |
| `5way_size` | 5 | Molecular size groups |
| `biological` | 4 | Non-polar, polar, positive, negative |

## Input formats

**Signal files** can be CSV or pickle. Each row represents one nanopore trace with columns for `run`, `channel`, amino acid label, and the signal data (as a list of floats or pre-segmented arrays).

**Barycenter files** are JSON with amino acid keys mapping to arrays of segment values.

**Profile files** are CSV with columns: `amino_acid`, `state`, `mean`, `std`. This lets you skip the barycenter-to-profile step and directly supply the emission parameters.

**Metadata files** (optional, JSON) specify which traces to analyze by run, channel, and amino acid.

## Key options

**Variance control** — The `--variance-mode` flag controls how emission variances are set. `barycenter` (default) derives variance from the barycenter segment spread, scaled by `--variance-scale`. `segment` uses empirical variances collected from actual signals.

**Per-amino-acid variance** — Supply `--variance-scale-file` with a CSV of `amino_acid,variance_scale` columns to use different variance scaling for each amino acid's model.

**Custom transitions** — Supply `--transition-file` with a JSON file of transition probabilities to override the defaults (forward, self-loop, skip, slip, insert, and end probabilities).

**Segmentation** — The `--seg-mode` flag selects the segmentation algorithm: `dynp` (dynamic programming, default), `pelt` (pruned exact linear time), or `set_window` (fixed window). Signals outside the `--min-signal-length` / `--max-signal-length` range are excluded.

**Backslip handling** — The `--backslip-mode` flag controls how segments that align to the same match state multiple times (due to backward translocation) are treated: `ignore` (concatenate all), `delete` (keep first occurrence), or `average` (DTW barycenter averaging, requires `dtaidistance`).

## Output

Results are written to `--output-dir` and include:

- `summary_*.csv` — Per-signal predictions with log-probabilities
- `results_*.json` — Full results including Viterbi paths and all model scores
- `metrics_*.json` — Accuracy and confusion matrix
- `hmm_profiles_*.csv` / `.json` — Learned emission parameters (when building from barycenters)
- `visualizations/` — Confusion matrices, pairwise accuracy heatmaps, per-trace HMM alignments, segment pileups, and backslip/skip distributions (disable with `--no-plots`)

## Project structure

```
vrhmm/
├── cli/
│   ├── args.py              # Argument parser
│   ├── main.py              # Entry point
│   └── runner.py            # Pipeline orchestration
├── core/
│   ├── hmm_builder.py       # Profile HMM construction
│   └── classifier.py        # HMM-based classification
├── io/
│   ├── loader.py            # Data loading (CSV, JSON, pickle)
│   └── writer.py            # Result serialization
├── processing/
│   ├── signal_processor.py  # Signal parsing, normalization, classification
│   └── segment_reorganizer.py  # HMM-based segment reordering
├── segmentation/
│   └── segmenter.py         # Signal segmentation and variance collection
├── utils/
│   ├── amino_acids.py       # Classification mode definitions
│   └── types.py             # Type definitions
├── visualization/
│   ├── classification_plots.py  # Confusion matrices, accuracy plots
│   ├── pairwise_matrix.py      # Pairwise AA discrimination analysis
│   └── signal_plots.py         # HMM alignment and segment visualizations
└── yahmm/                   # HMM library
```

## License

<!-- TODO: add license -->