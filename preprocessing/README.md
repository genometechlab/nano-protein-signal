# signal-cleaner

Modular multi-pass signal cleaning pipeline for nanopore trace data.

## What it does

Takes noisy nanopore signals and cleans them in up to **three configurable passes**:

| Pass | Purpose | Methods |
|------|---------|---------|
| **1 — Spike removal** | Detect and replace outlier spikes | `isolation` (Isolation Forest), `cwt_huber` (CWT + Huber regression), `hampel`, `ransac` |
| **2 — Smoothing** | Reduce broadband noise | `tv` (Total Variation), `lowpass` (Bessel/Butterworth), `bilateral`, `kalman`, `wavelet`, `none` |
| **3 — Refinement** | Catch residual spikes after smoothing | CWT + Huber with tighter thresholds (optional) |

Default pipeline: **Isolation Forest → TV Denoising → CWT+Huber** — the same parameters from the original nanoperpetrator config.

## Installation

```bash
git clone https://github.com/youruser/signal-cleaner.git
cd signal-cleaner
pip install -e .

# For development/testing:
pip install -e ".[dev]"
```

### Requirements
- Python ≥ 3.9
- numpy, scipy, scikit-learn, PyWavelets, numba, h5py, orjson, tqdm

## Quick start

### One-liner

```python
from signal_cleaner import clean_signal

cleaned = clean_signal(raw_signal)
```

### Full control

```python
from signal_cleaner import CleanerConfig, SignalCleaner, TraceData

config = CleanerConfig(
    first_pass_method="isolation",
    second_pass_method="tv",
    third_pass_cwt=True,
    contamination=0.1,
    weight=0.1,
)

cleaner = SignalCleaner(config)
trace = TraceData(raw_signal=my_array, metadata={"trace_id": "read_001"})
result = cleaner.process(trace)

print(result.metadata["noise_reduction"])  # e.g. 42.7 (percent)
print(result.cleaned_signal)               # numpy array
```

### Load from files

```python
from signal_cleaner import load_traces, clean_signal

# .fast5 (single or multi-read), directory of .fast5, or .json
traces = load_traces("my_data.fast5")
# traces = load_traces("reads_directory/")
# traces = load_traces("data.json", trace_keys=["read_001", "read_002"])

for t in traces:
    cleaned = clean_signal(t["signal"])
```

### Command line

```bash
# Default 3-pass cleaning
signal-cleaner data.json -o results/

# Custom passes, CSV output
signal-cleaner reads/ --first-pass cwt_huber --second-pass lowpass --no-third-pass --format csv

# Quick test on 20 traces
signal-cleaner data.fast5 --max-traces 20 -v
```

## JSON input formats

The loader accepts three JSON layouts:

**List of traces:**
```json
[
  {"trace_id": "t1", "raw": [1.2, 3.4, ...], "aa": "A"},
  {"trace_id": "t2", "raw": [5.6, 7.8, ...], "aa": "G"}
]
```

**Wrapped in a `traces` key:**
```json
{"traces": [{"trace_id": "t1", "raw": [...]}]}
```

**Column-oriented:**
```json
{
  "raw":     {"t1": [...], "t2": [...]},
  "aa":      {"t1": "A",  "t2": "G"},
  "channel": {"t1": 1,    "t2": 2}
}
```

## Configuration reference

All parameters with their defaults:

```python
CleanerConfig(
    # Pass 1 — Spike detection
    first_pass_method="isolation",   # cwt_huber | hampel | ransac | isolation
    sampling_rate=3012.0,
    spike_window_size=10,
    threshold_factor=0.15,           # CWT sensitivity
    dilation_size=3,                 # CWT spike mask expansion
    epsilon=2.0,                     # Huber robustness
    alpha=0.01,                      # Huber regularisation
    n_sigmas=3.0,                    # Hampel threshold
    contamination=0.1,               # Isolation Forest outlier fraction

    # Pass 2 — Smoothing
    second_pass_method="tv",         # lowpass | bilateral | tv | kalman | wavelet | none
    cutoff_freq=1500.0,              # Lowpass cutoff (Hz)
    filter_type="bessel",            # bessel | butterworth
    filter_order=2,
    spatial_sigma=2.0,               # Bilateral spatial
    range_sigma=5.0,                 # Bilateral range
    weight=0.1,                      # TV denoising strength
    tv_iterations=100,
    process_variance=1e-5,           # Kalman
    measurement_variance=0.01,       # Kalman

    # Pass 3 — CWT+Huber refinement
    third_pass_cwt=True,
    third_pass_threshold_factor=0.25,
    third_pass_window_size=5,
    third_pass_epsilon=1.0,
    third_pass_dilation_size=1,
)
```

## Project structure

```
signal-cleaner/
├── signal_cleaner/
│   ├── __init__.py          # Public API + clean_signal() convenience function
│   ├── cli.py               # Command-line interface
│   ├── core/
│   │   ├── config.py        # CleanerConfig dataclass
│   │   └── trace.py         # TraceData container
│   ├── processing/
│   │   ├── cleaner.py       # SignalCleaner orchestrator
│   │   └── filters.py       # All DSP functions (Numba-accelerated)
│   └── io/
│       └── loader.py        # .fast5 and .json loaders
├── tests/
│   └── test_cleaner.py
├── examples/
│   └── example_usage.py
├── pyproject.toml
└── README.md
```

## Running tests

```bash
pytest tests/ -v
```

## License

MIT
