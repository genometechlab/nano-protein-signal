# nano-extract

YY boundary detection and segment extraction for nanopore peptide signals.

Works on **nanoclean**-processed signals to find sustained YY dip regions, refine their boundaries, and extract labeled segments between dips.

## How it works

Nanopore peptide signals with the symmetric `SSGGYYGGSS...AA...SSGGYYGGSS` flanking structure produce **sustained current dips** at each YY position. This package:

1. **Detects YY dips** — finds contiguous regions where the cleaned signal drops below a threshold and stays low for ≥200 samples (not just brief spikes)
2. **Selects the top 5** — scores candidates by width × depth and keeps the best 5, matching the expected structure
3. **Refines boundaries** — searches around each dip edge to find the exact minimum sample
4. **Extracts segments** — cuts the signal between consecutive dip minimums, producing 4 inter-dip segments
5. **Labels segments** — optionally tags each segment with its amino acid from a run→peptide mapping

## Setup

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python ≥ 3.9.

```bash
git clone https://github.com/<your-org>/nano-extract.git
cd nano-extract
uv sync --all-extras
```

## Usage

### Python API

```python
from nano_extract import SegmentExtractor, ExtractionConfig

# Basic extraction (5 dips → 4 segments per trace)
extractor = SegmentExtractor()
segments_df = extractor.process_pickle("cleaned.pkl")

# With peptide labeling
cfg = ExtractionConfig(
    n_expected_dips=5,
    run_to_peptide={
        "20231124_run01_a": "HDKER",
        "20240130_run01_a": "GNQST",
    },
)
extractor = SegmentExtractor(cfg)
segments_df = extractor.process_pickle("cleaned.pkl")
```

### Command line

```bash
# Basic extraction
uv run nano-extract cleaned.pkl -o segments/

# With peptide mapping
uv run nano-extract cleaned.pkl --peptide-map peptides.json

# Custom parameters
uv run nano-extract cleaned.pkl --n-dips 5 --min-dip-width 200 --threshold-percentile 30
```

### Output format

The output DataFrame has one row per segment:

| Column | Description |
|--------|-------------|
| `run` | Run ID |
| `channel` | Channel number |
| `trace_id` | Trace identifier |
| `segment_index` | 0-3 (position between dips) |
| `start` | Start sample index in original signal |
| `end` | End sample index |
| `length` | Segment length in samples |
| `signal` | Extracted signal values (list) |
| `aa` | Amino acid label (if peptide mapping provided) |
| `left_dip_min` | Minimum current at left YY dip |
| `right_dip_min` | Minimum current at right YY dip |

## Configuration

```python
ExtractionConfig(
    # Smoothing (for detection only, doesn't modify the signal)
    smoothing_window=101,
    smoothing_polyorder=3,

    # Dip detection
    n_expected_dips=5,              # Fixed number of YY dips per trace
    dip_threshold_percentile=30.0,  # Below this = "in a dip"
    min_dip_width=200,              # Must stay low for this many samples

    # Boundary refinement
    refinement_padding=50,          # Search window around dip edges

    # Segments
    include_flanks=False,           # Also extract regions outside outermost dips
    min_segment_length=100,         # Discard tiny segments

    # Peptide labeling
    run_to_peptide={},              # {"run_id": "ABCDE"}
)
```

## Project structure

```
nano-extract/
├── nano_extract/
│   ├── __init__.py
│   ├── cli.py
│   ├── core/
│   │   └── config.py
│   ├── detection/
│   │   ├── dip_detector.py       # Sustained YY dip detection
│   │   └── boundary_refiner.py   # Edge refinement
│   ├── extraction/
│   │   └── extractor.py          # Main pipeline
│   └── test/
│       └── test_extraction.py
├── pyproject.toml
└── README.md
```

## Tests

```bash
uv run pytest -v
```

## License

MIT
