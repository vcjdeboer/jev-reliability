# Preregistration — Bayesian reliability validation of Jev

**Status: written before any data was collected.** The commit that introduces this
file predates every result. Deviations are recorded in §9, not silently folded in.

Author: V. de Boer (with Claude Opus 5)
Framework: OCRbayes — Zhang, Yuan, Keijer & de Boer, *PLOS ONE* 16(8):e0253926 (2021)

---

## 1. Motivation

TypeSafe documents Jev as returning calibrated judgments, and its guidance says to
"validate their performance in the target domain" — but supplies no method for doing
so. The cookbooks pick thresholds by assertion (`AUTO_ACCEPT = 0.8`).

OCRbayes supplies the missing method. It treats a noisy assay as a measurement with
*nested* sources of variation, decomposes that variation explicitly, and produces a
decision-ready posterior rather than a thresholded point estimate. We apply that
framework to Jev, treating **one Jev call as one assay**.

## 2. The measurement analogy, stated with its limits

OCRbayes models observed OCR as lognormal around a latent true value, with variance
components at measurement-cycle, well, and plate level.

For Jev, a `Score` answer returns a probability vector over K ordered levels plus
`score` — the probability-weighted expectation, in [1, K]. We take

> **y = `score`, rescaled to (0,1)**

as the continuous measurement. This is the direct analogue of OCR per 1k cells: a
continuous read of a latent quantity, observed with noise.

| OCRbayes facet | Jev analogue |
|---|---|
| measurement cycle (repeat reads of one well) | repeat API calls, byte-identical request |
| well (technical replicate, same plate) | semantically-null perturbation of the request |
| plate (biological replicate) | item (the text being judged) |

**Where the analogy holds.** Both are continuous measurements of a latent quantity
with nested variance components, and both decompose the same way.

**Where it fails, and what we do about it.** Two wells on a plate measure the *same*
quantity. Two paraphrases of a question may measure *slightly different constructs* —
so paraphrase disagreement is not cleanly "noise." We therefore split the middle facet
into two types and never pool them:

- **P-null (perturbation).** Changes that cannot change a correct answer: option
  order, trailing whitespace, key order in the state JSON, an irrelevant extra state
  field. Variance here is **unambiguously instrument noise**.
- **P-para (paraphrase).** Intended-equivalent rewording of instructions and criteria.
  Variance here is instrument noise **plus** construct drift.

The contrast **σ²_para − σ²_null identifies construct drift.** This is the
identification strategy, and it is the design's main contribution over "run it a few
times and look at the spread."

## 3. Research questions and estimands

| RQ | Tier | Estimand |
|---|---|---|
| RQ1 | 0 | Pr(identical probability vector \| byte-identical request) |
| RQ2 | 1 | σ²_null — variance from semantically-null perturbations |
| RQ3 | 2 | σ²_para − σ²_null — construct drift |
| RQ4 | 1 | Reliability G = σ²_item / (σ²_item + σ²_null + σ²_resid) |
| RQ5 | 3 | Calibration intercept and slope (**requires labels; out of scope here**) |

## 4. Design

Fully crossed: I items × F framings × R repeats.

- **Tier 0:** I = 5, F = 1, R = 20 → 100 calls.
- **Tier 1/2:** I = 30, F = 4 null + 4 para, R = 2 → 480 calls.

Items are drawn from a sampling frame declared in the run manifest *before*
collection. No item is swapped after seeing its result.

## 5. Statistical model

y_ijk ∈ (0,1), logit link:

```
logit(y_ijk) = mu + alpha_item[i] + beta_framing[j] + eps_ijk

alpha_item[i]    ~ Normal(0, sigma_item)
beta_framing[j]  ~ Normal(0, sigma_framing[type[j]])   # separate sigma for null vs para
eps_ijk          ~ Normal(0, sigma_resid)
```

**Priors — weakly informative, and here is why.** OCRbayes used informative priors
fitted by MLE from the OCR-stats reference dataset, and explicitly noted they must be
re-estimated per cell line. We have no equivalent reference corpus for Jev, and
inventing one would be circular. So:

```
mu             ~ Normal(0, 1.5)
sigma_*        ~ half-Normal(0, 1)
```

A prior predictive check is **required before fitting** to confirm these imply a
plausible spread of y.

## 6. Inference and diagnostics — pre-committed

- Stan via cmdstanr, 4 chains × 2000 iterations.
- **Report R̂ < 1.01 and ESS_bulk > 400 for every parameter**, or declare
  non-convergence and decline to interpret.
- Prior predictive check before fitting.
- Posterior predictive check: does the model reproduce the observed within-item spread
  of y?
- Prior sensitivity: refit with half-Normal(0, 0.5) and half-Normal(0, 2). Report
  whether the *ordering* of variance components changes.

## 7. Falsification — what would change our mind

Pre-committed, so results cannot be reinterpreted after the fact:

- **If Tier 0 shows determinism**, the measurement-cycle facet is empty. We drop
  `eps`, and the model becomes item + framing only. Committed to in advance.
- **If σ_null ≈ σ_para**, there is no measurable construct drift, and the paraphrase
  caution raised in design is unsupported by data. We report that against ourselves.
- **If σ_null ≥ σ_item**, Jev cannot separate these items reliably at all, and any
  ranking product built on it is unsound. This would kill the `jev-sort` idea, and we
  report it.

## 8. What this study cannot claim

- **Nothing about accuracy.** Reliability is not validity: a perfectly repeatable
  instrument can be repeatably wrong. Accuracy needs labels (Tier 3, out of scope).
- Nothing about other models, other question types, or other domains.
- No generalization beyond the declared item sampling frame.

## 8b. Study B — answerability (added 2026-09-19, before collection)

**Motivation.** Both this study and an independent one
([copyleftdev/jev-labs](https://github.com/copyleftdev/jev-labs)) hit the same
wall from opposite directions: *a stability gate detects jitter around a value; it
cannot detect that no value is warranted.* Our saturation result is the same fact
seen from the other side. TypeSafe's own guidance names the fix — "use a separate
presence judgment when it is independently useful" — but neither study built it.

**Design.** 20 items in two declared classes, fixed before collection:

- **judgeable** (n=10) — ordinary prose, where formality is a meaningful property.
- **unjudgeable** (n=10) — a bare number, a URL, a single word, punctuation, a
  base64 blob, a bare list. Text for which "how formal is this writing" has no
  defensible answer.

Each item gets the formality `Score` **and** an answerability `Noul` in the *same
request*, so both judgments see byte-identical state. Identity framing only,
4 repeats → 80 calls.

**Pre-registered predictions.**

- **H1** — mean `answerable` is lower for unjudgeable than for judgeable items.
- **H2, the one that matters** — `answerable` separates the two classes *better
  than `confidence` does*.

**Falsification, committed in advance.** If AUROC(confidence) ≈ AUROC(answerable),
then the Score's own confidence already carries the signal, the separate question
buys nothing, and the gap both studies identified is not real. We report that
result and do not build the gate.

The interesting failure mode to look for is the **dissociation**: high `confidence`
together with low `answerable` — Jev confidently rating the formality of something
that has no formality. That is precisely what a stability gate cannot catch.

## 9. Deviations from this plan

Recorded here with a reason, *before* any data was collected.

**D1 — "option order" dropped from the P-null set.** §2 originally listed option
order as a semantically-null perturbation. That is wrong for a **Score** question:
the criteria are *ordered levels*, so reversing them inverts the scale rather than
leaving meaning unchanged. It would have measured scale inversion and been reported
as instrument noise. Replaced with `quote_style` (straight → curly apostrophes).
Option-order reversal remains valid for a `Choice` question and can return if the
design is extended to Choice.

**D2 — Tier 0 item frame declared.** Five texts spanning the formality range, two of
them deliberately mid-scale (`i3`, `i4`). Rationale: determinism tested only at the
saturated ends of the scale (p ≈ 0.99) is trivially satisfied. The sensitive test is
where the distribution is genuinely split, so the frame must include such items or
the Tier 0 result is uninformative.

**D4 — `quote_style` excluded from the Tier 1/2 null set.** The transform replaces
straight apostrophes with curly ones, but the formality instructions contain **no
apostrophes**. It would therefore have produced a request byte-identical to
`identity` and entered the design as a silent duplicate, inflating the apparent
number of null framings while measuring nothing. The null set is `identity`,
`trailing_ws`, `key_reorder`, `irrelevant_field` — four, as §4 specifies.

**D5 — Tier 1/2 item frame stratified, not flat.** 30 items as 8 clear-casual, 14
mid-scale, 8 clear-formal. Rationale: Tier 0 established that measurement noise is
concentrated mid-scale and vanishes at saturation, so a frame sampled flat from a
natural corpus would be dominated by easy items and would **underestimate** σ_null.

*Consequence, stated up front:* deliberately over-sampling the hard region inflates
σ_null and deflates σ_item relative to a natural corpus. The resulting G is therefore
**not a corpus-wide reliability coefficient** — it is a conservative, hard-region
estimate. Per-stratum variance is reported alongside the pooled figure so the two are
not confused.

The five Tier 0 items are carried across (suffixed `_link_tier0`) as a linking subset,
so the two studies can be compared on common items.

**D9 — `key_reorder` was a no-op for the whole study, and it weakens one claim.**
Found after collection, by the no-op detector built during the redesign — not by
reading the code, which I had done repeatedly.

The transform did `Object.keys(s).sort().reverse()`. For the state `{text, task}`
that returns `["text", "task"]` — the original insertion order. The request was
therefore **byte-identical to `identity`**, confirmed by equal request hashes.

Consequence for Tier 1/2: the four declared P-null framings were really three
distinct requests plus a duplicate of `identity`. Two of the four null cells were
pure repeats of the same request, so the between-null-framing variance partly
measures repeat noise by construction.

**This weakens a claim made in the Tier 1/2 write-up.** σ_null (0.000100) matching
σ_repeat (0.000111) was presented as an internal check that the null framings are
genuinely null. It is **partly tautological**: half of those cells *were* repeats.
The direction of bias is toward the repeat-noise level — downward, if genuine null
effects exceed repeat noise, which `trailing_ws` and `irrelevant_field` should.

**RESOLVED — Tier 1/2 was re-collected** (`tier12-framing-fixed`, 480/480, no
no-op framings detected). The prediction held and was larger than expected:

| sd | biased | corrected |
|---|---|---|
| null perturbation | 0.0100 | **0.0204** |
| paraphrase | 0.0371 | 0.0347 |
| repeat | 0.0105 | 0.0102 |
| posterior s_null | 0.0105 [0.0011, 0.0188] | **0.0342 [0.0262, 0.0442]** |
| posterior drift | 0.00337 [0.00212, 0.00518] | **0.00218 [0.00072, 0.00405]** |

**Two claims are refuted, in both the descriptive and the Bayesian write-up.**
Each concluded that null perturbations plausibly have no effect beyond repeat
noise. Corrected, σ_null is roughly **twice** the repeat term and its posterior is
comfortably bounded away from zero: semantically-null changes are *not* free.

**The drift conclusion survives** — the interval still excludes zero — at a
smaller estimate, because the denominator had been depressed. The ratio falls from
3.7× to 1.70×, which lands beside copyleftdev/jev-labs' independently measured
1.74×; the corrected data reproduces their three-tier ordering (repeat <
perturbation < paraphrase) where the biased data collapsed two tiers into one.

The posterior predictive failure is **robust to the bug** — present in both
collections, and slightly stronger in the corrected one.

Fixed by reversing insertion order rather than sorting. The detector now ships in
the report and flags any framing whose request is byte-identical to another's.

**D10 — Student-t interaction fitted; RQ3 revised.** §5 specified a Normal
interaction. It failed its posterior predictive check in every collection
(ppp 0.964, 0.982), over-predicting how much a typical item moves, because two of
fourteen items are far more paraphrase-sensitive than the rest.

A Student-t interaction was therefore fitted — a scale mixture of normals, ν
slice-sampled under Gamma(2, 0.1). It **passes** (ppp 0.418) and converges
(R̂ 1.0011, ESS 3,582), with ν ≈ 2.48 [2.03, 3.68]: very heavy tails.

**Consequence for RQ3, stated plainly: under the model that fits, drift is no
longer established.** The interval moves from 0.00218 [0.00072, 0.00405] to
0.00043 [0.00000, 0.00107], and s_null and s_para overlap substantially. The drift
the Normal model reported was substantially carried by the two outlier items.

Paraphrase risk is therefore **concentrated rather than uniform** — most items
barely move under rewording, a minority move a great deal. That is a more useful
conclusion than the one it replaces, and it cannot be summarised by one number.

This is a posterior predictive comparison, not a formal model comparison; no LOO,
WAIC or Bayes factor is claimed.

**D6 — likelihood and scope changed; §5 as written cannot be fitted.** §5 specifies a
logit link on y ∈ (0,1). Tier 1/2 produced 13 items sitting at *exactly* 0.000 or
3.000, which map to the boundary where the logit is infinite. Worse, those items have
**zero** within-item variance — they are deterministic — so they carry no information
about any variance component and would break a homoscedastic likelihood.

Replaced with a Normal likelihood on the raw `score`, fitted to the **14 mid-stratum
items only** (224 observations). Justification: all measurable variance lives there,
and saturation is better reported as the descriptive fact it is than modelled as if it
were noise. The consequence is stated plainly: **the fitted components describe the
unsaturated region only** and do not generalise to a corpus containing saturated items.

The model also adds an item × framing interaction, which §5 omitted. The quantity of
interest for RQ2/RQ3 is how much a framing moves *a given item*, which is the
interaction — not the framing main effect. The descriptive pass conflated the two.

**D7 — rstan, not cmdstanr.** §6 specifies cmdstanr; it is not installed on this
machine and rstan is. Same sampler, same diagnostics. Iterations and chains unchanged.

**D3 — model pinned.** `jev-1.13.0` rather than `jev-latest`. An unpinned model would
confound version drift with measurement noise across the collection window. Every
observation also records the `model` string the API returned, so the pin is verifiable
after the fact rather than merely asserted.
