# Bayesian fit — variance components, and a model that fails its own check

**Fitted to the CORRECTED collection** (`data-fixed.csv`, from
`tier12-framing-fixed`). The earlier fit on the biased collection is superseded;
prereg **D9** explains why, and "What the bug changed" below records what moved.

**Data:** 224 observations, 14 mid-stratum items × 8 framings × 2 repeats.
**Model:** prereg §5 as amended by D6 — Normal likelihood on raw `score`,
unsaturated items only, with an item × framing interaction.
**Sampler:** stdlib Gibbs/slice, per D8. 4 chains × 60,000 iterations.

## The sampler was validated before it touched real data

It fits simulated data with known parameters and refuses to continue unless every
variance component is recovered inside its 95% interval. Recovery **PASS**, worst
R̂ 1.0122. Because the sampler is known to work, the misfit below is **model
misspecification, not a sampling artefact**.

The diagnostics also caught a defect in the first sampler: `mu` and the group
effects are identified only through their priors, so plain Gibbs crawled that
ridge (`mu` R̂ **3.36**). Likelihood-invariant joint shift moves dropped it to 1.01.

## Convergence gate — PASS

max R̂ **1.0013** (threshold < 1.01) · min ESS **3,921** (threshold > 400).

ESS improved by an order of magnitude over the biased fit (was 1,328). `s_null` is
no longer pinned against zero, where the sampler struggled.

## Posterior

| parameter | mean | 95% CI |
|---|---|---|
| mu | 1.6096 | [1.328, 1.888] |
| s_item | 0.5197 | [0.352, 0.791] |
| **s_para** | **0.0577** | **[0.0464, 0.0717]** |
| **s_null** | **0.0342** | **[0.0262, 0.0442]** |
| sigma_y (repeat) | 0.0213 | [0.0187, 0.0244] |
| s_fram (main effect) | 0.0081 | [0.0003, 0.0253] |
| **drift** = σ²_para − σ²_null | **0.00218** | **[0.00072, 0.00405]** |
| G | 0.9931 | [0.9858, 0.9976] |

**RQ3 under this model: the drift interval excludes zero.** But this model fails
its own predictive check, and **the better-fitting Student-t model dissolves that
result** — see "The Student-t model, and what it does to RQ3" at the end. Read the
two together; this number alone overstates the case.

**Prior sensitivity** (half-normal 0.5 / 1 / 2): s_null 0.0342–0.0344, s_para
0.0576–0.0579, G 0.9928–0.9933. Nothing is prior-driven.

## What the bug changed — a second claim refuted

| parameter | biased fit | corrected fit |
|---|---|---|
| s_null | 0.0105 **[0.0011, 0.0188]** | 0.0342 **[0.0262, 0.0442]** |
| s_para | 0.0588 | 0.0577 |
| sigma_y | 0.0222 | 0.0213 |
| drift | 0.00337 [0.00212, 0.00518] | 0.00218 [0.00072, 0.00405] |
| G | 0.9974 | 0.9931 |

The biased fit put s_null at [0.0011, 0.0188], nearly touching zero, and the
write-up concluded: *"the data are consistent with semantically-null perturbations
having no item-specific effect at all beyond repeat noise."*

**That is refuted.** Corrected, s_null is [0.0262, 0.0442] — comfortably bounded
away from zero and about 1.6× the repeat term. Null perturbations have a real,
item-specific effect. Two separate write-ups drew the opposite conclusion from a
duplicated framing.

The drift conclusion survives with a smaller estimate and a wider relative
interval, because the denominator was previously depressed.

## The posterior predictive still FAILS — and slightly harder

| | mean within-item sd |
|---|---|
| observed | 0.0408 |
| replicated 5% / 50% / 95% | 0.0424 / 0.0495 / 0.0590 |
| **ppp** | **0.982** (was 0.964) |

The observed statistic again falls below the 5th percentile of replicates. The
model over-predicts how much a typical item moves.

Per-item paraphrase range, sorted:

```
0.030 0.030 0.050 0.070 0.070 0.080 0.080 0.090 0.090 0.090 0.100 0.190 0.370 0.370
median 0.090   max 0.370   max/median 4.1x
```

**Paraphrase sensitivity is not homogeneous**, and the corrected data makes that
clearer (4.1× against 3.5×). Twelve items move a little; two move four times the
median. A Normal interaction assumes one common scale, stretches to cover the
outliers, and over-predicts everyone else.

The misfit is therefore **robust to the bug** — it appears in both collections,
and independently confirms the criticism raised when cross-checking against
copyleftdev/jev-labs: a single scalar paraphrase floor cannot describe this
distribution.

### What that does and does not invalidate

- **s_para must not be read as a per-item risk.** It is an average over a
  heterogeneous population — too loose for the two sensitive items, too tight for
  the other twelve.
- **The indicated next model is a Student-t interaction.** It has now been fitted,
  it passes this check, and it revises RQ3. See the next section — the drift
  conclusion does *not* survive it.

## The Student-t model, and what it does to RQ3

The failed predictive check indicated a heavy-tailed interaction, so one was
fitted: a scale mixture of normals (Andrews & Mallows 1974), with ν slice-sampled
under a Gamma(2, 0.1) prior (Juárez & Steel 2010). 4 chains × 50,000 iterations.

**Convergence: max R̂ 1.0011, min ESS 3,582 — PASS.**

| | Normal | Student-t |
|---|---|---|
| **ppp** (within-item sd) | **0.982 — fails** | **0.418 — passes** |
| s_null | 0.0342 [0.0262, 0.0442] | 0.0136 [0.0077, 0.0212] |
| s_para | 0.0577 [0.0464, 0.0717] | 0.0246 [0.0160, 0.0357] |
| sigma_y | 0.0213 | 0.0209 |
| s_item | 0.5197 | 0.5140 |
| **drift** | **0.00218 [0.00072, 0.00405]** | **0.00043 [0.00000, 0.00107]** |
| ν | — | **2.48 [2.03, 3.68]** |

The Student-t reproduces the observed within-item spread (observed 0.0408 against
a replicated median of 0.0394); the Normal cannot. ν ≈ 2.5 is extremely heavy
tailed — close to the point where the variance stops existing.

**RQ3 is revised.** Under the model that actually fits, **the drift interval
includes zero**. The credible intervals for s_null and s_para overlap
substantially. The evidence that paraphrasing moves a *typical* item more than a
semantically-null perturbation does is **not conclusive**.

The drift the Normal model declared established was substantially carried by the
two highly paraphrase-sensitive items. Forcing one common scale, the Normal spread
their influence across all fourteen and reported an average as if it were a
typical value.

**This does not mean paraphrasing is harmless — it means the risk is concentrated
rather than uniform.** Most items barely move under rewording; a minority move a
great deal. That is the practical finding, and it is more useful than the one it
replaces: you cannot characterise paraphrase sensitivity with a single number,
because the distribution has no typical member in the usual sense. It also
independently confirms, from model criticism, the criticism raised when
cross-checking against copyleftdev/jev-labs — that a scalar cohort floor cannot
describe this.

*Caveat on strength of evidence:* this is a posterior predictive comparison, not a
formal model comparison. No LOO or WAIC was computed, and no Bayes factor is
claimed. The statement supported is narrow and sufficient: the Normal model does
not reproduce this statistic of the data and the Student-t does.

## Limits

- Mid-stratum only (D6): these components describe the **unsaturated region**.
- 14 items, 8 framings, 2 repeats.
- Hand-written sampler because the machine's Stan toolchain is broken (D8);
  `model.stan` and `fit.R` are committed for reproduction under Stan, which has
  **not** been performed.
- Still nothing about accuracy.
