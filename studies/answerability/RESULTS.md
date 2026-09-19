# Study B results — answerability catches what confidence cannot

**Collected:** 2026-09-19T16:05:57Z · **Model:** `jev-1.13.0` · **Design:** 20 items
(10 judgeable prose, 10 unjudgeable), identity framing, 4 repeats = 80 calls,
80 collected, 0 failed. Score and answerability Noul asked in the **same request**,
so both judgments saw byte-identical state.

A first run was discarded: the Noul answer carries a `noul` field, not
`score`/`probabilities`, so the extractor returned null on all 80 rows. **No
answerability value was observed before the fix**, so the design was not altered in
response to data. The Noul also has no `confidence` field, which matches the
documented primitive — a Noul *is* a probability.

## Both pre-registered predictions hold

| metric | judgeable | unjudgeable | gap | AUROC |
|---|---|---|---|---|
| **answerable** | 0.9662 | 0.0542 | 0.9120 | **1.000** |
| confidence | 0.8928 | 0.3722 | 0.5205 | 0.940 |

**H1** — answerability is far lower for unjudgeable items. Confirmed, and the
separation is near-total: every judgeable item scores ≥ 0.9375, every unjudgeable
one ≤ 0.3075.

**H2** — answerability separates the classes *better* than confidence.
AUROC 1.000 vs 0.940, **Δ +0.060**. Perfect separation against merely good.
The pre-registered falsification condition (AUROC comparable → don't build it) was
**not** met, so the separate question earns its place.

## The dissociation, in one row

```
u03  "yes"   confidence = 0.982   score = 0.018   answerable = 0.020
```

Jev is **98% confident** that the single word "yes" is maximally casual writing, and
simultaneously reports a **2% chance** there is anything there to judge. A stability
gate sees 0.982, finds it rock-solid, and waves it through. The answerability
question catches it.

This is the failure mode named in copyleftdev/jev-labs — *the stability gate catches
jitter around a value, it cannot detect that no value is warranted* — reproduced
here as a single concrete row.

**It dissociates in both directions**, which is the stronger finding:

```
u03  unjudgeable, high confidence (0.982) -> false accept   (dangerous)
j09  judgeable,   low  confidence (0.500) -> false escalate (costly, safe)
```

`j09` is ordinary customer-service prose that Jev rates 0.978 answerable but only
0.500 confident. A confidence gate escalates it for no good reason. The two signals
are not redundant: **confidence asks "is the level determined?", answerability asks
"is there anything to level?"** Those come apart in both directions.

## What unjudgeable items score when nobody asks

Left to the Score alone, the unjudgeable items are not refused — they are rated:

| item | text | score |
|---|---|---|
| u05 | `3.14159 2.71828 1.61803 …` | 2.400 |
| u02 | `https://example.com/a/b?c=1&d=2` | 1.745 |
| u09 | `TODO` | 1.560 |
| u04 | `...` | 1.505 |

A bare list of constants is rated more formal than a release note. Without an
answerability question these values enter a ranking as if they meant something,
and nothing downstream can tell that they don't.

## Limits

- 20 items, one question, one model version. The unjudgeable class is deliberately
  extreme; the hard case is text that is *partially* judgeable, which this design
  does not probe.
- AUROC 1.000 on n=10 vs 10 means "no overlap in this sample", not "perfect
  detector". A larger and harder item set would be needed to estimate a real
  operating point, and no threshold is proposed here.
- Class labels are ours and were assigned by construction rather than by
  independent raters.
