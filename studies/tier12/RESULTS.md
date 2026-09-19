# Tier 1/2 results — variance decomposition and construct drift

**These are the results of the CORRECTED collection** (`tier12-framing-fixed`,
2026-09-19T17:12:09Z, 480/480, `jev-1.13.0`). The first collection
(`tier12-framing`) is retained in the repo but **superseded**: its `key_reorder`
framing was byte-identical to `identity` (prereg **D9**), so two of its four
null cells were pure repeats. See "What the bug changed" below — it reversed one
conclusion.

**Design:** 30 items × 8 framings (4 P-null, 4 P-para) × 2 repeats = 480 calls,
0 failed, no no-op framings detected. Scale is `score` ∈ [0, 3].

Descriptive moment decomposition (`analyze.py`), run before the Bayesian fit.
Cell means of 2 repeats carry `var_repeat/2` of sampling noise, so between-framing
variance is de-biased by subtracting it.

## Variance components

| component | variance | sd |
|---|---|---|
| item | 1.298700 | 1.1396 |
| framing — P-para | 0.001203 | 0.0347 |
| framing — P-null | 0.000417 | 0.0204 |
| repeat | 0.000104 | 0.0102 |

Three distinct tiers: **repeat 0.0102 < null 0.0204 < paraphrase 0.0347**.

## RQ3 — construct drift

σ²_para − σ²_null = **0.000786**. Paraphrase variance is 2.9× null variance, or
**1.70× on the sd scale**.

Modal-level movement: **0/30 items under P-null, 0/30 under P-para.** The decision
is robust across the board for this question, even though the underlying numbers
move.

## What the bug changed — one conclusion reversed

| component (sd) | biased collection | corrected |
|---|---|---|
| null perturbation | 0.0100 | **0.0204** |
| paraphrase | 0.0371 | 0.0347 |
| repeat | 0.0105 | 0.0102 |
| para / null ratio | 3.71× | **1.70×** |

**The reversed claim.** The first write-up reported σ_null (0.0100) matching
σ_repeat (0.0105) and offered it as an internal check that the null framings were
genuinely null — that a change which cannot alter a correct answer behaves like a
re-run. That was an artefact: half those cells *were* re-runs.

Corrected, **null perturbations produce about twice the variance of pure
repetition** (0.0204 vs 0.0102). Semantically-null changes are *not* free. Adding
an irrelevant field to the state, or trailing whitespace to the instructions,
moves the answer measurably more than simply asking the same question again.

That is a more useful finding than the one it replaces, and it is the opposite of
what was originally reported.

**What survives.** Drift is still established and still the largest framing term.
The ratio is smaller — 1.70× rather than 3.7× — because the earlier denominator
was depressed.

## Agreement with an independent study

copyleftdev/jev-labs reports three noise floors on the same model version:
identity 0.042 < question-reorder 0.059 < paraphrase 0.073, i.e. ratios of 1.40×
and 1.24×.

The corrected data reproduces that **three-tier ordering** — repeat < perturbation
< paraphrase. The biased data could not: it collapsed the perturbation tier onto
the repeat tier. Our para/null of 1.70× sits beside their para/identity of 1.74×.

Two studies, different designs, different questions, same structure.

## Per stratum

| stratum | n | var_item | var_null | var_para | var_rep |
|---|---|---|---|---|---|
| casual | 8 | 0.00001 | 0.00000 | 0.00000 | 0.00000 |
| mid | 14 | 0.20952 | 0.00089 | 0.00258 | 0.00022 |
| formal | 8 | 0.00001 | 0.00000 | 0.00000 | 0.00000 |

**D5 predicted the wrong direction**, and that stands. It argued over-sampling
mid-scale items would make G conservative; instead the saturated strata inflate
between-item variance and make the pooled G optimistic. The mid-stratum figure is
the meaningful one.

## What the corpus can be ranked into

Unchanged in kind from the first collection: the scale saturates hard, 13 of 30
items are exact ties (6 at 0.0000, 7 at 3.0000), and ranking within those groups
is arbitrary.

## Limits

- Descriptive moment estimates with no intervals; negative de-biased components
  floored at zero. Superseded by the Bayesian fit.
- 2 repeats per cell; 30 items; one question; one model version.
- **Still nothing about accuracy.**
