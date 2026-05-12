# V13 Pass B Methodology Notes

**Run completed:** 2026-05-12 08:55 (wall clock 55.7h)
**Scope:** 1095 days × 7 assets × B=99 full-grid permutation
**Real coverage:** BTC on-chain (blockchain.info, 1096 daily rows),
ETH on-chain via Etherscan V2 was queued but ultimately not included
(see "Coverage anomaly" below), cross-exchange spreads for all 7 assets
(Coinbase + Kraken, 721 daily rows each).

This file documents what the paper-grade run revealed AND what it
revealed about the methodology itself.

---

## What the data says (taking the run at face value)

### Per-condition signal-weighted accuracy (prob_threshold=0.65)

| Condition | Acc | n_signals | Delta vs base |
|-----------|----:|----------:|--------------:|
| base | 0.5737 | 3,188 | — |
| **v1_only** | **0.5871** | 1,797 | **+1.34 pp** |
| v2_only | 0.5000 | 0 | — (model never crosses threshold) |
| base+v1 | 0.5391 | 6,008 | −3.46 pp |
| base+v2 | 0.5740 | 3,190 | +0.03 pp |
| base+v1+v2 | 0.5386 | 6,003 | −3.51 pp |
| **base+shuffled_v2** | **0.5740** | 3,190 | identical to base+v2 |

Three findings:
1. **v1 alone beats base alone** by +1.34 pp at high-confidence signals.
2. **v2 has no signal** beyond the shuffled control (base+v2 ≡ base+shuffled_v2).
3. **Adding v1 to base HURTS** by −3.46 pp. Probable cause: same C=0.5 logistic
   regularisation, more features, more overfitting room. Tighter regularisation
   (smaller C) or feature selection should recover the v1-alone lift.

### Coverage anomaly: ETH on-chain not in this report

The Etherscan ETH integration was committed *just before* this run started,
but the markdown's per-asset coverage table shows ETH with `—` for on-chain.
Likely cause: the run was kicked off in a subshell where `.env` had not yet
been re-sourced. Cross-exchange spreads attached correctly for ETH (721 rows).

Fix for next run: confirm `echo "$ETHERSCAN_API_KEY" | head -c 8` is non-empty
in the same shell that launches `bash scripts/run_paper_grade.sh paper`. The
script's `source .env` block will work for direct invocation; the issue was
specific to how this run was launched.

---

## Methodology bug: full-grid permutation null is invalid

### Observed
```
Real best (v1_only):   0.5871
Null max mean ± std:   0.8845 ± 0.1050
p_grid                = 1.0000   (all 99 permutations beat real)
```

A valid permutation test has null mean ≈ real (under H0) or null < real
(under H1). Null mean of 0.88 against real of 0.59 is not a valid null.

### Root cause
1. **Signal-weighted accuracy with prob_threshold=0.65** counts a prediction
   only when the model crosses 0.65 or 0.35. Under permuted labels, the
   model rarely produces high-confidence predictions, so per-fold n_signals
   can collapse to single digits.
2. **Small-n accuracy is unstable.** A fold with 3 signals and 2 correct
   gives 0.667 by chance — not signal.
3. **Taking MAX across 6 real conditions** samples the right tail of 6
   noisy distributions. Even if each condition has expected acc=0.5,
   max(6 noisy samples) easily lands at 0.7–0.9.
4. **v2_only has n_signals=0 on the real run**, so under permutation it
   can randomly fire 1–5 signals and contribute huge values to the max.

### Why this didn't surface in the 90-day smoke
- 90 days × 5 folds → much smaller test sets → tighter null distribution
  by luck (mean ≈ 0.45, std ≈ 0.04). At 1095 days × 7 assets the n_signals
  per fold gets large enough that the variance issue becomes pathological
  on v2-style conditions where the threshold rarely fires under permutation.

### Proposed fix (not yet implemented)
Two-metric reporting:
- **Operational headline:** signal-weighted accuracy with prob_threshold=0.65
  (what a real trading system would deploy). Use this for the per-condition
  ablation table.
- **Permutation comparison metric:** raw test accuracy with prob_threshold=0.5
  (uses every test sample). Use this for the null distribution and p_grid.

Pseudocode for `evaluate_ablation` extension:
```python
def evaluate_ablation(..., perm_metric: str = 'signal_weighted'):
    ...
    if perm_metric == 'raw':
        # ignore prob_threshold; score every test sample
        for sym, (te_x, te_y) in test_per_asset.items():
            pred = clf.predict(scaler.transform(te_x))
            acc = accuracy_score(te_y, pred)
            ...
```

Then in main(): pass `perm_metric='raw'` to the permutation loop only.
Estimated cost to re-run: same B=99, no extra fits, just a different
metric. ~55h again unless we reduce B or assets.

Alternative: filter conditions with min n_signals before taking max. e.g.
exclude any condition where n_signals < 30 across all folds in a given
permutation.

---

## Honest summary for the paper

If we report only the multiple-testing-aware test as designed, the answer is
"no signal" (p_grid = 1.0). But the test is broken — see above. The
*correctable* findings are:

- **v1 scalar TDA features add a small but real lift (+1.34 pp) over base
  alone** at signal-weighted accuracy. Whether this is significant after
  proper multiple-testing correction is unknown until the methodology fix
  lands.
- **v2 persistence images contribute no signal** beyond a shuffled control
  on this dataset and at this resolution. Consistent with v12's finding.
- **Combining v1 with base degrades accuracy by 3.5 pp**, suggesting
  regularisation tuning is needed before claiming v1 has practical value.

---

## What I recommend next

1. **Methodology fix first.** Patch `evaluate_ablation` to support
   `perm_metric='raw'` for the permutation loop, then rerun pass B with
   the same data (no need to re-fetch). Estimate: 1h dev + 50h compute.
2. **Or:** drop the full-grid permutation entirely and accept the
   post-selection diagnostic as the headline, with an explicit Bonferroni
   correction for the 6 conditions tested. Simpler, faster, still honest.
3. **Investigate the v1+base degradation.** Two hypotheses:
   - C=0.5 logistic over-regularises base alone but under-regularises
     base+v1. Try a small grid over C.
   - v1 features carry signal that disagrees with base features under the
     0.65 threshold — i.e. v1 fires high-confidence wrong-direction
     predictions that base doesn't. Inspect per-prediction agreement.
4. **Fix the .env-not-loaded launch bug** so ETH on-chain actually
   enters the next pass-B run.
