import { assert, assertAlmostEquals, assertEquals } from "jsr:@std/assert@1";
import {
  analyse,
  decisionOf,
  type Grid,
  type Observation,
  splitRhat,
  val,
} from "./jev_reliability_report.ts";

let hashSeed = 0;

function obs(p: Partial<Observation> & { item: string }): Observation {
  return {
    framing: "identity",
    framingType: "null",
    repeat: 1,
    requestHash: `h${hashSeed++}`,
    score: 1,
    confidence: 0.9,
    probabilities: { "0": 0.05, "1": 0.9, "2": 0.03, "3": 0.02 },
    probsHash: "p",
    model: "jev-test",
    ...p,
  } as Observation;
}

function grid(observations: Observation[], overrides: Partial<Grid> = {}): Grid {
  return {
    run: "t",
    collectedAt: "2026-09-19T00:00:00Z",
    model: "jev-test",
    question: { instructions: "q", criteria: ["a", "b", "c", "d"] },
    design: {
      items: new Set(observations.map((o) => o.item)).size,
      framings: [{ id: "identity", type: "null" }],
      repeats: 2,
      totalCalls: observations.length,
    },
    observations,
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// Backwards compatibility with grids collected before the fields existed
// ---------------------------------------------------------------------------

Deno.test("val falls back to score on older grids", () => {
  assertEquals(val(obs({ item: "a", score: 2.4 })), 2.4);
  assertEquals(val(obs({ item: "a", score: 2.4, value: 1.1 })), 1.1);
});

Deno.test("decisionOf falls back to the argmax on older grids", () => {
  assertEquals(decisionOf(obs({ item: "a" })), "1");
  assertEquals(decisionOf(obs({ item: "a", decision: "yes" })), "yes");
  assertEquals(
    decisionOf(obs({ item: "a", probabilities: null, decision: null })),
    null,
  );
});

// ---------------------------------------------------------------------------
// Resolution: identical values cannot be ranked
// ---------------------------------------------------------------------------

Deno.test("exact ties are reported, near-ties are not", () => {
  const a = analyse(
    grid([
      obs({ item: "x", score: 3 }),
      obs({ item: "y", score: 3 }),
      obs({ item: "z", score: 3 }),
      obs({ item: "w", score: 1.5 }),
    ]),
    1000,
  );
  assertEquals(a.exactTies.length, 1);
  assertEquals(a.exactTies[0].items, ["x", "y", "z"]);
  assertEquals(a.distinctMeans, 2);
  assert(
    a.warnings.some((w) => w.includes("EXACT ties")),
    "expected a tie warning",
  );
});

// ---------------------------------------------------------------------------
// A framing that sent an identical request measured nothing
// ---------------------------------------------------------------------------

Deno.test("framings with an identical request hash are flagged as no-ops", () => {
  // This is the check that caught a control of ours that was secretly the
  // control it was supposed to differ from.
  const a = analyse(
    grid([
      obs({ item: "x", framing: "identity", requestHash: "SAME" }),
      obs({ item: "x", framing: "key_reorder", requestHash: "SAME" }),
      obs({ item: "x", framing: "trailing_ws", requestHash: "OTHER" }),
    ]),
    1000,
  );
  assertEquals(a.noopFramings.length, 1);
  assert(a.noopFramings[0].includes("key_reorder"));
  assert(a.warnings.some((w) => w.includes("BYTE-IDENTICAL")));
});

Deno.test("distinct request hashes produce no no-op warning", () => {
  const a = analyse(
    grid([
      obs({ item: "x", framing: "identity", requestHash: "A" }),
      obs({ item: "x", framing: "key_reorder", requestHash: "B" }),
    ]),
    1000,
  );
  assertEquals(a.noopFramings.length, 0);
});

// ---------------------------------------------------------------------------
// Flip rates must nest against one reference decision
// ---------------------------------------------------------------------------

Deno.test("flip rates are measured against the item's overall modal decision", () => {
  // Two calls say "yes", one says "no". Using a per-pool mode made every
  // subset look internally consistent and the rates stopped nesting.
  const a = analyse(
    grid([
      obs({ item: "x", framingType: "null", decision: "yes", value: 1 }),
      obs({ item: "x", framingType: "null", decision: "yes", value: 1 }),
      obs({
        item: "x",
        framing: "para_1",
        framingType: "para",
        decision: "no",
        value: 0,
      }),
    ]),
    1000,
  );
  assertEquals(a.flipStable, 0);
  assertEquals(a.flipPara, 1);
  assertAlmostEquals(a.flipAll, 1 / 3, 1e-9);
  assertEquals(a.flippedItems, ["x"]);
});

Deno.test("a stable item reports no flips anywhere", () => {
  const a = analyse(
    grid([
      obs({ item: "x", decision: "yes", value: 1 }),
      obs({ item: "x", decision: "yes", value: 1 }),
    ]),
    1000,
  );
  assertEquals(a.flipAll, 0);
  assertEquals(a.flippedItems.length, 0);
});

// ---------------------------------------------------------------------------
// Answerability is reported on its own, not only when confidence is high
// ---------------------------------------------------------------------------

Deno.test("low answerability is reported regardless of confidence", () => {
  const a = analyse(
    grid([
      obs({ item: "prose", answerable: 0.97, confidence: 0.9 }),
      obs({ item: "confident_junk", answerable: 0.02, confidence: 0.98 }),
      obs({ item: "unsure_junk", answerable: 0.04, confidence: 0.30 }),
    ]),
    1000,
  );
  // Both junk items are unanswerable, even though only one is confident.
  assertEquals(a.answerability?.lowAnswerable.length, 2);
  assertEquals(a.answerability?.lowAnswerHighConf.length, 1);
  assertEquals(a.answerability?.lowAnswerHighConf[0].item, "confident_junk");
  assert(a.warnings.some((w) => w.includes("score low on answerability")));
});

Deno.test("a grid with no answerability question says so", () => {
  const a = analyse(grid([obs({ item: "x" }), obs({ item: "y", score: 2 })]), 1000);
  assertEquals(a.answerability, null);
  assert(a.warnings.some((w) => w.includes("No answerability question")));
});

// ---------------------------------------------------------------------------
// Threshold stability
// ---------------------------------------------------------------------------

Deno.test("confidence straddling a common cut-off is flagged", () => {
  const a = analyse(
    grid([
      obs({ item: "x", confidence: 0.46 }),
      obs({ item: "x", confidence: 0.53 }),
      obs({ item: "y", confidence: 0.91 }),
      obs({ item: "y", confidence: 0.93 }),
    ]),
    1000,
  );
  assertEquals(a.thresholdFlappers.length, 1);
  assertEquals(a.thresholdFlappers[0].item, "x");
});

// ---------------------------------------------------------------------------
// Convergence diagnostic
// ---------------------------------------------------------------------------

Deno.test("splitRhat is ~1 for chains drawn from one distribution", () => {
  const chains = [0, 1, 2, 3].map((k) =>
    Array.from({ length: 400 }, (_, i) => Math.sin(i + k * 0.5))
  );
  const r = splitRhat(chains);
  assert(r < 1.05, `expected Rhat near 1, got ${r}`);
});

Deno.test("splitRhat is large for chains that disagree", () => {
  const chains = [
    Array.from({ length: 400 }, (_, i) => Math.sin(i) + 0),
    Array.from({ length: 400 }, (_, i) => Math.sin(i) + 50),
  ];
  const r = splitRhat(chains);
  assert(r > 1.5, `expected a large Rhat, got ${r}`);
});
