"""
signal_cleaner.core.trace
==========================
Lightweight container for a single signal trace and its cleaning results.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TraceData:
    """Container that travels through every pipeline stage.

    At minimum you supply ``raw_signal``.  Each cleaning stage populates
    additional fields.
    """

    raw_signal: np.ndarray
    """Original unmodified signal (never mutated by the pipeline)."""

    cleaned_signal: Optional[np.ndarray] = None
    """Signal after all cleaning passes."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Arbitrary key-value store for trace provenance, metrics, etc."""

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Flat dictionary suitable for a DataFrame row."""
        d: Dict[str, Any] = {
            "raw": self.raw_signal.tolist(),
            "cleaned": (
                self.cleaned_signal.tolist()
                if self.cleaned_signal is not None
                else None
            ),
        }
        d.update(self.metadata)
        return d

    @property
    def noise_reduction_pct(self) -> Optional[float]:
        """Percent noise reduction (stored by the cleaner in metadata)."""
        return self.metadata.get("noise_reduction")

    def __repr__(self) -> str:
        n = len(self.raw_signal)
        status = "cleaned" if self.cleaned_signal is not None else "raw"
        return f"TraceData(n={n}, status={status})"
