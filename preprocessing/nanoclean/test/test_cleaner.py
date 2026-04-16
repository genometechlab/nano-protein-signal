"""
Tests for the signal cleaning pipeline.

Run with:  pytest tests/ -v
"""

import numpy as np
import pytest

from nanoclean import clean_signal, CleanerConfig, SignalCleaner, TraceData
from nanoclean.processing import filters


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def clean_sine():
    """1-second sine wave at 50 Hz, sampled at 3012 Hz."""
    t = np.linspace(0, 1.0, 3012)
    return np.sin(2 * np.pi * 50 * t)


@pytest.fixture
def spiked_sine(clean_sine):
    """Sine wave with 20 random spike injections."""
    rng = np.random.default_rng(42)
    signal = clean_sine.copy()
    spike_idx = rng.choice(len(signal), size=20, replace=False)
    signal[spike_idx] += rng.normal(0, 10, size=20)
    return signal, spike_idx


# =====================================================================
# Config validation
# =====================================================================

class TestConfig:
    def test_defaults_are_valid(self):
        cfg = CleanerConfig()
        assert cfg.first_pass_method == "isolation"
        assert cfg.second_pass_method == "tv"
        assert cfg.third_pass_cwt is True

    def test_rejects_bad_first_pass(self):
        with pytest.raises(ValueError, match="first_pass_method"):
            CleanerConfig(first_pass_method="bogus")

    def test_rejects_bad_second_pass(self):
        with pytest.raises(ValueError, match="second_pass_method"):
            CleanerConfig(second_pass_method="bogus")

    def test_to_filter_params_keys(self):
        params = CleanerConfig().to_filter_params()
        assert "sampling_rate" in params
        assert "window_size" in params
        assert "contamination" in params


# =====================================================================
# Filter primitives
# =====================================================================

class TestFilters:
    def test_compute_mad(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        mad = filters.compute_mad(data)
        assert mad > 0

    def test_hampel_removes_spike(self):
        data = np.zeros(100)
        data[50] = 50.0  # obvious spike
        cleaned, mask = filters.hampel_filter_numba(data, 5, 3.0)
        assert mask[50]
        assert abs(cleaned[50]) < 5.0

    def test_tv_denoising_reduces_noise(self):
        rng = np.random.default_rng(0)
        clean = np.ones(200) * 5.0
        noisy = clean + rng.normal(0, 0.5, 200)
        denoised = filters.tv_denoise_numba(noisy, weight=0.1, n_iter=50)
        assert np.std(denoised) < np.std(noisy)

    def test_bilateral_preserves_edges(self):
        # Step function
        data = np.concatenate([np.zeros(100), np.ones(100) * 10])
        filtered = filters.bilateral_filter_numba(data, 2.0, 1.0)
        # Edge should still be sharp-ish
        assert filtered[50] < 1.0
        assert filtered[150] > 9.0

    def test_kalman_smooths(self):
        rng = np.random.default_rng(1)
        noisy = np.sin(np.linspace(0, 4 * np.pi, 500)) + rng.normal(0, 0.3, 500)
        filtered = filters.kalman_filter_numba(noisy, 1e-5, 0.01)
        assert np.std(np.diff(filtered)) < np.std(np.diff(noisy))

    def test_cwt_detects_spikes(self, spiked_sine):
        signal, true_spikes = spiked_sine
        mask = filters.detect_spikes_cwt(signal, 3012.0, "mexh", 0.15)
        # Should flag at least some of the injected spikes
        detected_at_true = mask[true_spikes].sum()
        assert detected_at_true > 0

    def test_isolation_forest_detects_spikes(self, spiked_sine):
        signal, _ = spiked_sine
        _, mask = filters.detect_and_remove_spikes_isolation(
            signal, contamination=0.1, window_size=5
        )
        assert mask.sum() > 0

    def test_lowpass(self, clean_sine):
        noisy = clean_sine + np.random.default_rng(2).normal(0, 0.1, len(clean_sine))
        filtered = filters.apply_lowpass(noisy, 3012.0, 500.0, "bessel", 2)
        # High-freq noise should be reduced
        assert np.std(filtered - clean_sine) < np.std(noisy - clean_sine)

    def test_wavelet_denoising(self):
        rng = np.random.default_rng(3)
        clean = np.sin(np.linspace(0, 2 * np.pi, 512))
        noisy = clean + rng.normal(0, 0.3, 512)
        denoised = filters.apply_wavelet_denoising(noisy)
        assert np.std(denoised - clean) < np.std(noisy - clean)


# =====================================================================
# Full pipeline
# =====================================================================

class TestPipeline:
    def test_default_pipeline(self, spiked_sine):
        signal, _ = spiked_sine
        cleaned = clean_signal(signal)
        assert len(cleaned) == len(signal)
        # Should reduce noise
        assert np.std(np.diff(cleaned)) < np.std(np.diff(signal))

    def test_all_first_pass_methods(self, spiked_sine):
        signal, _ = spiked_sine
        for method in ["cwt_huber", "hampel", "ransac", "isolation"]:
            result = clean_signal(signal, first_pass=method, second_pass="none", third_pass=False)
            assert len(result) == len(signal)
            assert not np.any(np.isnan(result))

    def test_all_second_pass_methods(self, spiked_sine):
        signal, _ = spiked_sine
        for method in ["lowpass", "bilateral", "tv", "kalman", "wavelet", "none"]:
            result = clean_signal(
                signal, first_pass="hampel", second_pass=method, third_pass=False
            )
            assert len(result) == len(signal)

    def test_three_pass_pipeline(self, spiked_sine):
        """The exact combo requested: isolation → tv → cwt_huber."""
        signal, _ = spiked_sine
        cleaned = clean_signal(
            signal,
            first_pass="isolation",
            second_pass="tv",
            third_pass=True,
        )
        assert len(cleaned) == len(signal)
        assert np.std(np.diff(cleaned)) < np.std(np.diff(signal))

    def test_cleaner_records_metrics(self, spiked_sine):
        signal, _ = spiked_sine
        cfg = CleanerConfig()
        cleaner = SignalCleaner(cfg)
        trace = TraceData(raw_signal=signal)
        cleaner.process(trace)

        assert "noise_reduction" in trace.metadata
        assert "processing_method" in trace.metadata
        assert trace.metadata["processing_method"] == "isolation+tv+cwt_huber"

    def test_trace_to_dict(self, spiked_sine):
        signal, _ = spiked_sine
        trace = TraceData(raw_signal=signal, metadata={"trace_id": "test_001"})
        SignalCleaner().process(trace)
        d = trace.to_dict()
        assert "raw" in d
        assert "cleaned" in d
        assert d["trace_id"] == "test_001"

    def test_fallback_on_bad_signal(self):
        """Pipeline should not crash on degenerate input."""
        # All zeros — some methods may struggle
        signal = np.zeros(100)
        cleaned = clean_signal(signal)
        assert len(cleaned) == 100

    def test_config_overrides_via_convenience(self, spiked_sine):
        signal, _ = spiked_sine
        cleaned = clean_signal(signal, contamination=0.2, weight=0.05)
        assert len(cleaned) == len(signal)


# =====================================================================
# Batch processing
# =====================================================================

class TestBatchCleaner:
    def _make_traces(self, n=20):
        """Create n fake trace dicts."""
        rng = np.random.default_rng(42)
        traces = []
        for i in range(n):
            t = np.linspace(0, 1.0, 3012)
            signal = np.sin(2 * np.pi * 50 * t) + rng.normal(0, 0.3, len(t))
            signal[rng.choice(len(signal), 5, replace=False)] += rng.normal(0, 10, 5)
            traces.append({
                "trace_id": f"trace_{i:03d}",
                "signal": signal,
            })
        return traces

    def test_batch_processes_all_traces(self):
        from nanoclean import BatchCleaner
        traces = self._make_traces(10)
        with BatchCleaner(n_workers=2) as bc:
            df = bc.process_traces(traces, show_progress=False)
        assert len(df) == 10
        assert "cleaned" in df.columns
        assert "noise_reduction" in df.columns

    def test_batch_sequential_fallback(self):
        """Small datasets should use sequential processing."""
        from nanoclean import BatchCleaner
        traces = self._make_traces(3)
        with BatchCleaner(n_workers=2) as bc:
            df = bc.process_traces(traces, show_progress=False)
        assert len(df) == 3

    def test_batch_handles_empty_input(self):
        from nanoclean import BatchCleaner
        with BatchCleaner() as bc:
            df = bc.process_traces([], show_progress=False)
        assert df.empty

    def test_batch_skips_bad_traces(self):
        from nanoclean import BatchCleaner
        traces = self._make_traces(5)
        traces[2]["signal"] = None  # bad trace
        with BatchCleaner(n_workers=2) as bc:
            df = bc.process_traces(traces, show_progress=False)
        assert len(df) == 4  # 5 minus the bad one

    def test_batch_custom_batch_size(self):
        from nanoclean import BatchCleaner
        traces = self._make_traces(12)
        with BatchCleaner(n_workers=2, batch_size=3) as bc:
            df = bc.process_traces(traces, show_progress=False)
        assert len(df) == 12

    def test_batch_context_manager_cleans_up(self):
        from nanoclean import BatchCleaner
        bc = BatchCleaner(n_workers=2)
        traces = self._make_traces(8)
        bc.process_traces(traces, show_progress=False)
        assert bc._pool is not None
        bc.shutdown()
        assert bc._pool is None