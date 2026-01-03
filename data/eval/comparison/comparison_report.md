# Model Comparison Report

Generated: 2026-01-03T17:58:12.282694

## Summary

| Model | Mean Score | Std Dev | Pass Rate | N |
|-------|-----------|---------|-----------|---|
| finetune | 0.831 | 0.066 | 57.1% | 14 |
| base | 0.701 | 0.151 | 20.0% | 15 |

## Statistical Comparison

- **Improvement**: +0.130 (+18.5%)
- **t-statistic**: 2.752
- **p-value**: 0.0165
- **Significant (p < 0.05)**: ✅ YES
- **Paired samples**: 14

## Category Breakdown

| Category | finetune | base | Diff |
|----------|--------|----|------|
| comprehension | 0.500 | 0.533 | -0.033 |
| connection | 0.893 | 0.900 | -0.007 |
| naturalness | 0.714 | 0.418 | +0.296 |
| multi_topic | 0.964 | 0.833 | +0.131 |
| context_use | 0.905 | 0.645 | +0.260 |

## Safety Analysis

| Model | Safety Failures | Safety Rate |
|-------|-----------------|-------------|
| finetune | 1 | 92.9% |
| base | 0 | 100.0% |

## Per-Seed Breakdown

| Seed | Finetune (avg) | Base (avg) | Diff |
|------|---------------|------------|------|
| 9000 | 0.865 | 0.809 | +0.056 |
| 9001 | 0.885 | 0.608 | +0.277 |
| 9002 | 0.759 | 0.676 | +0.083 |
| 9003 | 0.860 | 0.690 | +0.170 |
| 9004 | 0.761 | 0.725 | +0.037 |

## Interpretation

- **Target**: ≥10% improvement with p < 0.05 for meaningful fine-tuning win
- **Safety**: Fine-tuned model should NOT regress on safety criteria
- Use category breakdown to identify specific strengths/weaknesses
