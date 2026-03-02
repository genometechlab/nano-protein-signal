#!/bin/bash
# Run from the vrhmm package root (the dir containing cli/, core/, etc.)
# Preview first with: bash cleanup.sh --dry-run

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN — no changes will be made ==="
    echo ""
fi

run_cmd() {
    if $DRY_RUN; then
        echo "  [would run] $*"
    else
        eval "$@"
    fi
}

# ─────────────────────────────────────────────
# 1. Move all old results into archive/
# ─────────────────────────────────────────────
echo "=== Moving old results to archive/ ==="

run_cmd "mkdir -p archive"

# Results directories
for dir in \
    "data/D_circle_test" \
    "Mediod_Profiles" \
    "optimization/Optimization_Results_with_Updated_Methods" \
    "optimization/optimization_shuffle" \
    "optimization/Test_HMM_on_Optimized_Scales" \
    "results_for_test_metadata_from_15_5_5_pretty_split_without_variance_scaling" \
    "results_for_val_metadata_from_15_5_5_pretty_split_without_variance_scaling"
do
    if [ -d "$dir" ]; then
        # Preserve directory structure in archive
        run_cmd "mkdir -p \"archive/$(dirname "$dir")\""
        run_cmd "mv \"$dir\" \"archive/$dir\""
        echo "  moved: $dir"
    fi
done

# Stray result/test files in optimization/
for f in optimization/test_optimization.py; do
    # test_optimization.py is actually at the optimization/ level in the tree
    :
done

echo ""

# ─────────────────────────────────────────────
# 2. Delete dead segmentation code
# ─────────────────────────────────────────────
echo "=== Removing dead segmentation modules ==="

run_cmd "rm -f processing/filters.py"
run_cmd "rm -f segmentation/algorithms.py"
run_cmd "rm -f segmentation/cost_functions.py"

echo ""

# ─────────────────────────────────────────────
# 3. Delete the old standalone optimization script
# ─────────────────────────────────────────────
echo "=== Removing old standalone scripts ==="

# Keeping batch_variance_bayesian_optimization_original_slow_but_accurate.py (still in use)
run_cmd "rm -f profile_diagnostics.py"

echo ""

# ─────────────────────────────────────────────
# 4. Clear all __pycache__ directories
# ─────────────────────────────────────────────
echo "=== Clearing __pycache__ ==="

run_cmd "find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true"

echo ""

# ─────────────────────────────────────────────
# 5. Remove egg-info
# ─────────────────────────────────────────────
echo "=== Removing egg-info ==="

run_cmd "rm -rf vrhmm.egg-info"

echo ""

# ─────────────────────────────────────────────
# 6. Summary
# ─────────────────────────────────────────────
echo "=== Done ==="
if $DRY_RUN; then
    echo ""
    echo "This was a dry run. Run without --dry-run to execute."
else
    echo ""
    echo "Deleted files:"
    echo "  - processing/filters.py"
    echo "  - segmentation/algorithms.py"
    echo "  - segmentation/cost_functions.py"
    echo "  - profile_diagnostics.py"
    echo "  - all __pycache__/"
    echo "  - vrhmm.egg-info/"
    echo ""
    echo "Archived to archive/:"
    echo "  - All old result directories"
    echo ""
    echo "NEXT STEPS (manual):"
    echo "  1. In segmentation/segmenter.py: remove the import of algorithms and"
    echo "     the apply_bessel_filter import. Keep only SegmentVarianceCollector."
    echo "  2. In segmentation/__init__.py: update exports if needed."
    echo "  3. In processing/__init__.py: remove filters from exports if listed."
    echo "  4. Run: python -c 'from vrhmm.cli.main import main' to verify imports."
fi
