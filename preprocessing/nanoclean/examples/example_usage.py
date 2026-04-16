"""
example_usage.py
================
Demonstrates the three main ways to use the signal_cleaner package.
"""

import numpy as np

# ------------------------------------------------------------------
# 1. Quickest path — one-liner convenience function
# ------------------------------------------------------------------

from nanoclean import clean_signal

# Generate a fake noisy signal with spikes
rng = np.random.default_rng(42)
t = np.linspace(0, 1.0, 3012)
raw = np.sin(2 * np.pi * 50 * t) + rng.normal(0, 0.3, len(t))
raw[rng.choice(len(raw), 15, replace=False)] += rng.normal(0, 10, 15)  # spikes

# Clean with defaults (isolation → tv → cwt_huber)
cleaned = clean_signal(raw)
print(f"Noise std  raw: {np.std(np.diff(raw)):.4f}")
print(f"Noise std  cln: {np.std(np.diff(cleaned)):.4f}")


# ------------------------------------------------------------------
# 2. Full control — config + cleaner + trace objects
# ------------------------------------------------------------------

from nanoclean import CleanerConfig, SignalCleaner, TraceData

config = CleanerConfig(
    first_pass_method="isolation",
    second_pass_method="tv",
    third_pass_cwt=True,
    contamination=0.1,
    weight=0.1,
)

cleaner = SignalCleaner(config)
trace = TraceData(
    raw_signal=raw,
    metadata={"trace_id": "demo_001", "channel": 42},
)
result = cleaner.process(trace)

print(f"\nNoise reduction: {result.metadata['noise_reduction']:.1f}%")
print(f"Method used:     {result.metadata['processing_method']}")


# ------------------------------------------------------------------
# 3. Load from files
# ------------------------------------------------------------------

from nanoclean import load_traces

# From JSON (uncomment when you have a real file):
# traces = load_traces("data.json")

# From fast5:
# traces = load_traces("reads/")

# From a directory of fast5 files, only first 10:
# traces = load_traces("reads/", max_traces=10)

# Process loaded traces:
# for t in traces:
#     cleaned = clean_signal(t["signal"])


# ------------------------------------------------------------------
# 4. Quick plot (requires matplotlib)
# ------------------------------------------------------------------

try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(t, raw, linewidth=0.5, alpha=0.8)
    axes[0].set_title("Raw signal (with spikes + noise)")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(t, result.cleaned_signal, linewidth=0.5, color="C1")
    axes[1].set_title(f"Cleaned ({result.metadata['processing_method']})")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig("example_output.png", dpi=150)
    print("\nPlot saved to example_output.png")
except ImportError:
    print("\nInstall matplotlib to see a plot: pip install matplotlib")
