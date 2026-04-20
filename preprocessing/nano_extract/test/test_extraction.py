"""
Tests for the nano-extract pipeline.

Run with:  pytest nano_extract/test/ -v
"""

import numpy as np
import pytest

from nano_extract import (
    ExtractionConfig,
    DipDetector,
    DipRegion,
    BoundaryRefiner,
    SegmentExtractor,
)


# =====================================================================
# Fixtures
# =====================================================================

def _make_signal_with_dips(n_dips=5, dip_width=300, segment_width=1500, dip_depth=8.0):
    """Create a synthetic signal with known YY dip positions."""
    rng = np.random.default_rng(42)
    baseline = 50.0
    parts = []

    for i in range(n_dips):
        if i == 0:
            seg = np.full(segment_width, baseline) + rng.normal(0, 0.3, segment_width)
            parts.append(seg)

        dip = np.full(dip_width, baseline - dip_depth) + rng.normal(0, 0.3, dip_width)
        parts.append(dip)

        seg = np.full(segment_width, baseline) + rng.normal(0, 0.3, segment_width)
        parts.append(seg)

    return np.concatenate(parts)


@pytest.fixture
def synthetic_signal():
    """Signal with 5 known dips and 4 segments between them."""
    return _make_signal_with_dips(n_dips=5)


@pytest.fixture
def config():
    return ExtractionConfig(n_expected_dips=5, min_dip_width=100)


# =====================================================================
# Config
# =====================================================================

class TestConfig:
    def test_defaults(self):
        cfg = ExtractionConfig()
        assert cfg.n_expected_dips == 5
        assert cfg.min_dip_width == 200
        assert cfg.dip_threshold_percentile == 30.0
        assert not hasattr(cfg, 'merge_gap')

    def test_custom(self):
        cfg = ExtractionConfig(n_expected_dips=7, min_dip_width=100)
        assert cfg.n_expected_dips == 7

    def test_peptide_mapping(self):
        cfg = ExtractionConfig(run_to_peptide={"run01": "ABCD"})
        assert cfg.run_to_peptide["run01"] == "ABCD"


# =====================================================================
# Dip Detector
# =====================================================================

class TestDipDetector:
    def test_detects_correct_count(self, synthetic_signal, config):
        detector = DipDetector(config)
        dips = detector.detect(synthetic_signal)
        assert len(dips) == 5

    def test_dips_are_sorted(self, synthetic_signal, config):
        detector = DipDetector(config)
        dips = detector.detect(synthetic_signal)
        starts = [d.start for d in dips]
        assert starts == sorted(starts)

    def test_dip_regions_are_correct_type(self, synthetic_signal, config):
        detector = DipDetector(config)
        dips = detector.detect(synthetic_signal)
        for dip in dips:
            assert isinstance(dip, DipRegion)
            assert dip.width >= config.min_dip_width
            assert dip.depth > 0

    def test_detect_with_metadata(self, synthetic_signal, config):
        detector = DipDetector(config)
        result = detector.detect_with_metadata(synthetic_signal)
        assert all(k in result for k in ["dips", "smoothed", "threshold", "success"])
        assert result["success"] is True
        assert len(result["smoothed"]) == len(synthetic_signal)

    def test_trims_excess_dips(self, config):
        signal = _make_signal_with_dips(n_dips=8, dip_width=300, segment_width=800)
        config.n_expected_dips = 5
        detector = DipDetector(config)
        dips = detector.detect(signal)
        assert len(dips) == 5

    def test_warns_on_too_few(self, config):
        signal = _make_signal_with_dips(n_dips=2, dip_width=300, segment_width=800)
        config.n_expected_dips = 5
        detector = DipDetector(config)
        dips = detector.detect(signal)
        assert len(dips) < 5

    def test_empty_signal(self, config):
        detector = DipDetector(config)
        dips = detector.detect(np.array([50.0] * 100))
        assert len(dips) == 0

    def test_closely_spaced_dips_not_merged(self):
        """Closely spaced dips should remain separate — no merging."""
        rng = np.random.default_rng(42)
        baseline = 50.0
        depth = 8.0
        parts = []
        # Create 5 dips with short segments between some of them
        widths = [1500, 400, 1500, 400, 1500]  # alternating long/short segments
        for i in range(5):
            if i == 0:
                parts.append(np.full(1000, baseline) + rng.normal(0, 0.3, 1000))
            parts.append(np.full(300, baseline - depth) + rng.normal(0, 0.3, 300))
            parts.append(np.full(widths[i], baseline) + rng.normal(0, 0.3, widths[i]))

        signal = np.concatenate(parts)
        cfg = ExtractionConfig(n_expected_dips=5, min_dip_width=100)
        detector = DipDetector(cfg)
        dips = detector.detect(signal)
        assert len(dips) == 5


# =====================================================================
# Boundary Refiner
# =====================================================================

class TestBoundaryRefiner:
    def test_refine_returns_results(self, synthetic_signal, config):
        detector = DipDetector(config)
        dips = detector.detect(synthetic_signal)
        refiner = BoundaryRefiner(config)
        refined = refiner.refine(synthetic_signal, dips)
        assert len(refined) == len(dips)
        for r in refined:
            assert "left_idx" in r
            assert "right_idx" in r
            assert "min_idx" in r

    def test_segment_boundaries(self, synthetic_signal, config):
        detector = DipDetector(config)
        dips = detector.detect(synthetic_signal)
        refiner = BoundaryRefiner(config)
        boundaries = refiner.get_segment_boundaries(synthetic_signal, dips)
        # 5 dips → 4 segments
        assert len(boundaries) == 4
        for start, end in boundaries:
            assert start < end

    def test_boundaries_are_non_overlapping(self, synthetic_signal, config):
        detector = DipDetector(config)
        dips = detector.detect(synthetic_signal)
        refiner = BoundaryRefiner(config)
        boundaries = refiner.get_segment_boundaries(synthetic_signal, dips)
        for i in range(len(boundaries) - 1):
            assert boundaries[i][1] <= boundaries[i + 1][0]

    def test_include_flanks(self, synthetic_signal):
        cfg = ExtractionConfig(n_expected_dips=5, min_dip_width=100, include_flanks=True)
        detector = DipDetector(cfg)
        dips = detector.detect(synthetic_signal)
        refiner = BoundaryRefiner(cfg)
        boundaries = refiner.get_segment_boundaries(synthetic_signal, dips)
        # With flanks: left_flank + 4 segments + right_flank = 6
        assert len(boundaries) >= 5


# =====================================================================
# Segment Extractor
# =====================================================================

class TestSegmentExtractor:
    def test_extract_single(self, synthetic_signal, config):
        extractor = SegmentExtractor(config)
        segments = extractor.extract_single(
            synthetic_signal, run="test_run", channel=1, trace_id="t1"
        )
        assert len(segments) == 4
        for seg in segments:
            assert "run" in seg
            assert "channel" in seg
            assert "signal" in seg
            assert "segment_index" in seg
            assert seg["run"] == "test_run"

    def test_peptide_labeling(self, synthetic_signal):
        cfg = ExtractionConfig(
            n_expected_dips=5,
            min_dip_width=100,
            run_to_peptide={"test_run": "ABCD"},
        )
        extractor = SegmentExtractor(cfg)
        segments = extractor.extract_single(synthetic_signal, run="test_run")
        assert segments[0]["aa"] == "A"
        assert segments[1]["aa"] == "B"
        assert segments[2]["aa"] == "C"
        assert segments[3]["aa"] == "D"

    def test_no_peptide_gives_none(self, synthetic_signal, config):
        extractor = SegmentExtractor(config)
        segments = extractor.extract_single(synthetic_signal, run="unknown_run")
        for seg in segments:
            assert seg["aa"] is None

    def test_process_dataframe(self, synthetic_signal, config):
        import pandas as pd

        df = pd.DataFrame([
            {"cleaned": synthetic_signal.tolist(), "run": "r1", "channel": 1, "trace_id": "t1"},
            {"cleaned": synthetic_signal.tolist(), "run": "r1", "channel": 2, "trace_id": "t2"},
        ])
        extractor = SegmentExtractor(config)
        result = extractor.process_dataframe(df, show_progress=False)
        assert len(result) == 8  # 2 traces × 4 segments
        assert "segment_index" in result.columns
        assert result["trace_id"].nunique() == 2

    def test_handles_bad_signal(self, config):
        extractor = SegmentExtractor(config)
        segments = extractor.extract_single(np.zeros(50))
        assert len(segments) == 0

    def test_segment_signals_are_correct_length(self, synthetic_signal, config):
        extractor = SegmentExtractor(config)
        segments = extractor.extract_single(synthetic_signal)
        for seg in segments:
            expected_len = seg["end"] - seg["start"]
            assert len(seg["signal"]) == expected_len
            assert seg["length"] == expected_len
