# Tier 0 results — is Jev deterministic?

**Collected:** 2026-09-19T15:49:25Z · **Model:** `jev-1.13.0` (pinned; the model
string returned by the API matches the pin on all 100 observations, so the pin is
verified rather than asserted) · **Design:** 5 items × 1 framing (`identity`) × 20
repeats = 100 calls, 100 collected, 0 failed.

Every call in a given item's block sent a **byte-identical request body** (verified by
`requestHash`, one distinct hash per item).

## Headline

**No. Jev is not deterministic — but only where the distribution is unsaturated, and
the modal answer never moved.**

| item | n | distinct distributions | distinct `score` | score range | sd |
|---|---|---|---|---|---|
| i1_casual | 20 | **1** | 1 | 0.0000 – 0.0000 | 0.00000 |
| i2_relaxed | 20 | 6 | 5 | 0.6700 – 0.7100 | 0.01195 |
| i3_mid_ambiguous | 20 | 8 | 7 | 1.8400 – 1.9000 | 0.01639 |
| i4_mid_ambiguous | 20 | **16** | 10 | 1.1700 – 1.3000 | 0.03433 |
| i5_formal | 20 | **1** | 1 | 3.0000 – 3.0000 | 0.00000 |

The noisiest item returned 16 different probability vectors in 20 identical calls:

```
{'0': 0.17, '1': 0.49, '2': 0.30, '3': 0.04}
{'0': 0.15, '1': 0.55, '2': 0.27, '3': 0.03}
{'0': 0.13, '1': 0.52, '2': 0.31, '3': 0.04}
{'0': 0.15, '1': 0.53, '2': 0.29, '3': 0.03}
{'0': 0.16, '1': 0.54, '2': 0.27, '3': 0.03}
```

## The design decision that saved the result

Prereg **D2** required mid-scale items on the grounds that determinism tested only at
saturated ends would be trivially satisfied. That is exactly what happened: the two
saturated items (i1 at 0.00, i5 at 3.00, both confidence 1.00) are **perfectly
deterministic**, 1 distinct distribution in 20 calls.

A study using only clear-cut texts — the obvious thing to do — would have concluded
"Jev is deterministic" and been wrong.

## The decision is stable even though the distribution is not

**The argmax never flipped.** Exactly one modal level was observed per item across all
20 repeats, including the noisiest. So a Choice-style routing decision on this question
is robust to the jitter.

What is *not* robust is a threshold near a cut point:

| item | confidence range | top-probability range |
|---|---|---|
| i2_relaxed | 0.66 – 0.71 | 0.67 – 0.71 |
| i3_mid_ambiguous | 0.82 – 0.86 | 0.82 – 0.86 |
| i4_mid_ambiguous | **0.46 – 0.53** | 0.49 – 0.55 |

i4's confidence **straddles 0.50 on identical input**. A gate written as
`confidence >= 0.5` would flip between runs on the same item, with nothing changed.
This is a concrete argument against fixed cookbook cutoffs (`AUTO_ACCEPT = 0.8` and
similar): items sitting within roughly ±0.05 of a threshold will flap, and the flapping
is invisible unless you replicate.

## Consequence for the model — pre-committed branch resolved

§7 committed in advance: *"If Tier 0 shows determinism, the measurement-cycle facet is
empty. We drop `eps`."*

It did not. **`eps` stays.** The three-level structure — repeat → framing → item — is
identifiable, and Tier 1/2 is worth running. The residual facet is real but small
relative to between-item spread (within-item sd ≤ 0.034 against between-item gaps of
roughly 0.6 on the same 0–3 scale), which is an encouraging early sign for the
reliability coefficient — to be estimated properly in Tier 1, not from these numbers.

## Limits of this result

- **n = 20 per item, 5 items.** The sd figures are crude point estimates with no
  interval. They indicate presence and rough magnitude of noise, nothing more.
- **Reported probabilities are quantised to 2 decimal places.** Observed differences
  span several quantisation steps so the jitter is real, but its fine structure is
  below the reported resolution.
- **One question, one model version, one domain.** Nothing here generalises to other
  questions, `jev-latest`, Choice/Noul primitives, or other text types.
- **Nothing about accuracy.** This measures repeatability only. Every item could be
  repeatably mis-rated and this study could not tell.
