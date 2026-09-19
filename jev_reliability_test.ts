import { assert, assertEquals, assertThrows } from "jsr:@std/assert@1";
import {
  extractNoul,
  type Framing,
  interpret,
  NULL_FRAMINGS,
  resolveApiKey,
} from "./jev_reliability.ts";

const framing = (id: string): Framing => {
  const f = NULL_FRAMINGS.find((x) => x.id === id);
  if (!f) throw new Error(`no framing ${id}`);
  return f;
};

// ---------------------------------------------------------------------------
// The API key check has to be able to reject
// ---------------------------------------------------------------------------

Deno.test("resolveApiKey returns a real key", () => {
  assertEquals(resolveApiKey({ apiKey: "sk-live" } as never), "sk-live");
});

Deno.test("resolveApiKey rejects an unresolved vault expression", () => {
  // A non-empty string, so a naive presence test would pass while the vault
  // is still empty. This is the bug the check exists to catch.
  assertThrows(
    () => resolveApiKey({ apiKey: '${{ vault.get("v", "K") }}' } as never),
    Error,
    "unresolved expression",
  );
});

// ---------------------------------------------------------------------------
// Answer interpretation, one shape per primitive
// ---------------------------------------------------------------------------

Deno.test("interpret: noul yields P(yes) and a thresholded decision", () => {
  assertEquals(interpret({ type: "noul", noul: 0.72 }, "noul", 0.5), {
    value: 0.72,
    decision: "yes",
  });
  assertEquals(interpret({ type: "noul", noul: 0.72 }, "noul", 0.8), {
    value: 0.72,
    decision: "no",
  });
});

Deno.test("interpret: noul decision flips exactly at the threshold", () => {
  // >= threshold counts as yes; the boundary must not be ambiguous.
  assertEquals(
    interpret({ type: "noul", noul: 0.5 }, "noul", 0.5).decision,
    "yes",
  );
  assertEquals(
    interpret({ type: "noul", noul: 0.4999 }, "noul", 0.5).decision,
    "no",
  );
});

Deno.test("interpret: choice yields the selected option and its probability", () => {
  const a = {
    type: "choice",
    choice: "escalate",
    probabilities: { escalate: 0.61, ignore: 0.39 },
  };
  assertEquals(interpret(a, "choice", 0.5), {
    value: 0.61,
    decision: "escalate",
  });
});

Deno.test("interpret: score yields the weighted value and the modal level", () => {
  const a = {
    type: "score",
    score: 1.24,
    probabilities: { "0": 0.15, "1": 0.55, "2": 0.27, "3": 0.03 },
  };
  assertEquals(interpret(a, "score", 0.5), { value: 1.24, decision: "1" });
});

Deno.test("interpret: a missing answer yields nulls rather than throwing", () => {
  assertEquals(interpret({ type: "score" }, "score", 0.5), {
    value: null,
    decision: null,
  });
});

Deno.test("extractNoul prefers the documented noul field", () => {
  // The response carries `noul`, not `score`. Reading the wrong field once
  // cost us a whole collection.
  assertEquals(extractNoul({ type: "noul", noul: 0.96 }), 0.96);
  assertEquals(extractNoul({ type: "noul", noul: 0, score: 0.5 }), 0);
  assertEquals(extractNoul({ type: "noul" }), null);
});

// ---------------------------------------------------------------------------
// Framings — regression tests for a control that was not a control
// ---------------------------------------------------------------------------

Deno.test("key_reorder actually changes the serialised field order", () => {
  // Regression: this transform used to sort the keys and then reverse them,
  // which for a two-field object returns the original order. It was
  // byte-identical to identity for an entire 480-call collection.
  const state = { text: "hello", task: "judge it" };
  const out = framing("key_reorder").state!(state);
  assert(
    JSON.stringify(out) !== JSON.stringify(state),
    "key_reorder produced a byte-identical request",
  );
  assertEquals(Object.keys(out), ["task", "text"]);
});

Deno.test("key_reorder preserves every field and value", () => {
  const state = { text: "hello", task: "judge it", extra: 1 };
  const out = framing("key_reorder").state!(state);
  assertEquals(
    Object.fromEntries(Object.entries(out).sort()),
    Object.fromEntries(Object.entries(state).sort()),
  );
});

Deno.test("trailing_ws changes the instructions without changing meaning", () => {
  const i = "Rate the formality of this writing.";
  const out = framing("trailing_ws").instructions!(i);
  assert(out !== i);
  assertEquals(out.trim(), i);
});

Deno.test("irrelevant_field adds a field and leaves the rest alone", () => {
  const state = { text: "hello", task: "judge it" };
  const out = framing("irrelevant_field").state!(state);
  assertEquals(out.text, "hello");
  assertEquals(out.task, "judge it");
  assert("record_locator" in out);
});

Deno.test("quote_style is a no-op without apostrophes — known and detected", () => {
  // Documented, not accidental: the report catches this by request hash
  // rather than pretending the framing did something.
  const plain = "Rate the formality of this writing.";
  assertEquals(framing("quote_style").instructions!(plain), plain);

  const withQuote = "Rate the author's formality.";
  assert(framing("quote_style").instructions!(withQuote) !== withQuote);
});

Deno.test("no null framing reverses ordered criteria", () => {
  // Reversing an ordered Score scale inverts its meaning, so it must not be
  // in the null set however much it looks like a harmless permutation.
  for (const f of NULL_FRAMINGS) {
    assertEquals(f.type, "null");
    const touched = f.instructions?.("a. low b. high") ?? "";
    assert(
      !touched.includes("b. high a. low"),
      `${f.id} reordered the criteria`,
    );
  }
});
