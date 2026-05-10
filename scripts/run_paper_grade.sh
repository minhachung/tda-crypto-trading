#!/usr/bin/env bash
# Paper-grade v13 run — multiple-testing aware, real external data, B=99 minimum.
#
# Two passes:
#   PASS A: smoke check — small B, full grid, fast.
#   PASS B: paper-grade — 1095d × 7 assets × B=99 full-grid (multiple-testing aware).
#
# Estimated runtimes on Apple Silicon (M-series):
#   PASS A: ~5–10 minutes
#   PASS B: ~6–10 hours (run overnight)
#
# Usage:
#   bash scripts/run_paper_grade.sh smoke
#   bash scripts/run_paper_grade.sh paper
#   bash scripts/run_paper_grade.sh both     # (default)

set -euo pipefail

cd "$(dirname "$0")/.."

mode="${1:-both}"

PASS_A_LOG="results/v13_pass_a_smoke.log"
PASS_B_LOG="results/v13_pass_b_paper.log"
mkdir -p results

run_smoke() {
    echo "=========================================================="
    echo " PASS A — smoke (90d × BTC,ETH × B=19, full-grid)"
    echo "=========================================================="
    python3 examples/run_validation_v13.py 90 19 \
        --symbols BTC,ETH \
        --horizon 72 \
        --window-size 20 \
        --n-splits 5 \
        --embargo 0 \
        --full-grid \
        --quick \
        2>&1 | tee "$PASS_A_LOG"
    cp results/V13_COMPREHENSIVE.md results/V13_PASS_A_SMOKE.md
    echo "✓ Pass A done. Log: $PASS_A_LOG"
}

run_paper() {
    echo "=========================================================="
    echo " PASS B — paper-grade (1095d × 7 assets × B=99, full-grid)"
    echo "=========================================================="
    echo " Estimated runtime: 6–10h. Safe to run overnight."
    echo ""
    python3 examples/run_validation_v13.py 1095 99 \
        --symbols BTC,ETH,SOL,ADA,DOT,LINK,AVAX \
        --horizon 72 \
        --window-size 20 \
        --n-splits 5 \
        --embargo 24 \
        --full-grid \
        2>&1 | tee "$PASS_B_LOG"
    cp results/V13_COMPREHENSIVE.md results/V13_PASS_B_PAPER.md
    echo "✓ Pass B done. Log: $PASS_B_LOG"
}

case "$mode" in
    smoke) run_smoke ;;
    paper) run_paper ;;
    both)  run_smoke; echo ""; run_paper ;;
    *)
        echo "Usage: $0 {smoke|paper|both}"
        exit 1
        ;;
esac
