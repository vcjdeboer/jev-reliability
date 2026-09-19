# @vcjdeboer/jev-reliability

**Is this Jev question safe to build on?** A pre-flight check for
[TypeSafe](https://docs.typesafe.ai) System One questions, built on
[swamp](https://github.com/swamp-club/swamp).

Jev turns text into numbers. Before you put one of those numbers behind an `if`,
you want to know four things:

1. **Repeatability** — ask the identical question twice, do you get the identical answer?
2. **Framing** — does rewording the question move the answer, and by how much more than plain noise?
3. **Resolution** — can this question actually tell your items apart, or does the scale saturate into ties?
4. **Answerability** — is it confidently rating things that have nothing to rate?

This extension runs the measurements and a report turns them into a verdict.

It works on all three System One primitives, because each returns a distribution
plus a decision:

| primitive | typical use | value analysed | decision |
| --- | --- | --- | --- |
| `score` | rating, ranking | probability-weighted level | modal level |
| `noul` | **gates** | P(yes) | yes/no at your threshold |
| `choice` | **routers** | P(selected option) | the selected option |

## Install

```sh
swamp extension pull @vcjdeboer/jev-reliability
```

> **Not on the registry yet.** Until it is, clone this repo and copy
> `jev_reliability.ts` into your own repo's `extensions/models/` and
> `jev_reliability_report.ts` into `extensions/reports/` — swamp picks up local
> extensions from those directories.

You need a TypeSafe API key. Store it in a vault rather than an env var:

```sh
swamp vault create local_encryption typesafe
swamp vault put typesafe TYPESAFE_API_KEY      # prompts; nothing lands in shell history
```

## Use it

```sh
swamp model create @vcjdeboer/jev-reliability myquestion
```

Then set the key reference and **pin the model version** in the model definition —
an unpinned model confounds version drift with measurement noise:

```yaml
globalArguments:
  apiKey: '${{ vault.get("typesafe", "TYPESAFE_API_KEY") }}'
  model: jev-1.13.0
reports:
  require:
    - '@vcjdeboer/jev-reliability-report'
```

Write an input file describing your question and some items:

```json
{
  "name": "is-this-ticket-urgent",
  "questionType": "noul",
  "threshold": 0.5,
  "instructions": "Does this message require a response from a human today?",
  "criteriaBool": {
    "true": "A person must act on this today.",
    "false": "This can wait, needs no reply, or is automated noise."
  },
  "paraphrases": [
    "Is same-day human attention actually needed here?",
    "Must a person deal with this message before the end of today?"
  ],
  "answerability": "Does this text contain an actual message, with enough content to judge whether it needs a reply?",
  "repeats": 2,
  "items": [
    { "id": "t1", "text": "Production checkout has been failing for 40 minutes..." }
  ]
}
```

See the cost before you spend it:

```sh
swamp model method run @vcjdeboer/jev-reliability check myquestion \
  --input-file myquestion.json --input dryRun=true
```

Run it, then read the verdict:

```sh
swamp model method run @vcjdeboer/jev-reliability check myquestion \
  --input-file myquestion.json

swamp report get @vcjdeboer/jev-reliability-report --model myquestion
```

### Methods

| Method | Does |
| --- | --- |
| `check` | the one to use — question plus items, design defaulted, prints the call count first, `dryRun` stops before spending |
| `replicate` | the low-level escape hatch: every design knob explicit |

Both record one row per (item, framing, repeat) with the full distribution, the
decision, the confidence, and a SHA-256 of the exact request body.

## What the report tells you

The headline is a **decision flip rate** — if you reran this, how often would the
decision your code acts on come out differently?

```
### Headline — decision flip rate

| condition              | flip rate |
|------------------------|-----------|
| same request, repeated | 0.0%      |
| question reworded      | 12.5%     |
| overall                | 3.6%      |
```

Then a verdict, e.g.:

```
- 13 of 30 items are EXACT ties — they cannot be ranked against each other at all.
- Rewording the question moves answers 1.7x more than semantically-null changes do.
- 9 item(s) have confidence straddling a common cut-off on IDENTICAL input.
- 1 framing(s) produced a BYTE-IDENTICAL request to another and measured nothing.
```

Plus a nested Bayesian variance decomposition (repeat → framing → item, after
OCRbayes[^1]) fitted by a Gibbs/slice sampler written in TypeScript — **no Stan, R
or Python required**. It **withholds the posterior when the convergence gate
fails** (R̂ < 1.01, ESS > 400) rather than printing numbers you should not read.

`studies/example-report.txt` is a full rendered report over five real grids.

## Why paraphrases matter, and why the tool checks yours

The largest term is almost always **question phrasing**, not random noise. You
supply the paraphrases, so the tool does not take them on trust: it hashes every
request body and refuses to count a "perturbation" that produced a byte-identical
request.

That check immediately caught a bug in our own preregistered study — a framing
that looked like it reordered JSON keys but returned the original order, making it
identical to the control for an entire collection. It had survived repeated code
review. See `docs/preregistration.md`, deviation **D9**.

## What this cannot tell you

**Nothing about accuracy.** It measures whether a question is *consistent*, never
whether it is *right*. A perfectly repeatable question can be repeatably wrong.
Checking correctness needs labelled data and a different study.

## The data

Everything in `studies/` is real, collected against `jev-1.13.0`:

| study | calls | what it establishes |
| --- | --- | --- |
| `tier0` | 100 | Jev is not deterministic — but only off-saturation |
| `tier12` | 480 | variance decomposition; paraphrase ≫ perturbation ≫ repeat |
| `answerability` | 80 | a separate presence question catches what confidence cannot |
| `noul` | 112 | the same machinery on a gate-shaped question |

Design, predictions and falsification criteria were fixed **before** collection in
`docs/preregistration.md`, and every later change is logged there as a numbered
deviation — including the ones that went against us.

## References

[^1]: Zhang, Y., Yuan, F., Keijer, J., & de Boer, V.C.J. (2021). *OCRbayes: A
Bayesian hierarchical modeling framework for Seahorse extracellular flux oxygen
consumption rate data analysis.* PLOS ONE 16(8): e0253926.
<https://doi.org/10.1371/journal.pone.0253926>

## Tests

```sh
deno test --allow-env jev_reliability_test.ts jev_reliability_report_test.ts
```

26 tests, no network. They cover answer interpretation for all three
primitives, the API-key check refusing an unresolved vault expression, the
no-op framing detection, the flip-rate reference decision, low-answerability
reporting, and a regression test for a control that was once byte-identical to
doing nothing.

## Licence

Code MIT, writing and data CC BY 4.0. See `LICENSE.md`.
