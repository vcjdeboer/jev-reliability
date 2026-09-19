/**
 * @vcjdeboer/jev-reliability — measure the instrument, not the text.
 *
 * A Jev call is an assay. This extension runs a fully crossed
 * item x framing x repeat grid against the TypeSafe System One API and records
 * every raw answer, so the resulting variance can be decomposed with a nested
 * Bayesian measurement model (see docs/jev-bayesian-validation-prereg.md).
 *
 * Design follows OCRbayes (Zhang, Yuan, Keijer & de Boer, PLOS ONE 2021):
 * measurement cycle -> well -> plate becomes repeat -> framing -> item.
 *
 * Works on all three System One primitives, because each returns a
 * distribution plus a decision:
 *
 *   score   value = probability-weighted level   decision = modal level
 *   noul    value = P(yes)                       decision = yes/no at threshold
 *   choice  value = P(selected option)           decision = selected option
 *
 * Methods:
 * - **check**     — the one to use. Question + items, everything else
 *                   defaulted; prints the call count first and can dry-run.
 * - **replicate** — the low-level escape hatch: every design knob explicit.
 *
 * @module
 */
import { z } from "npm:zod@4";

const API_KEY_ENV = "TYPESAFE_API_KEY";
const DEFAULT_BASE_URL = "https://api.typesafe.ai";
const DEFAULT_MODEL = "jev-latest";
const RETRYABLE = new Set([408, 429, 500, 502, 503, 504, 529]);

// ---------------------------------------------------------------------------
// Global arguments
// ---------------------------------------------------------------------------

const GlobalArgsSchema = z.object({
  apiKey: z.string().min(1).meta({ sensitive: true }).optional().describe(
    "TypeSafe API key. Prefer a vault reference: " +
      '${{ vault.get("typesafe", "TYPESAFE_API_KEY") }}. ' +
      `Falls back to the ${API_KEY_ENV} env var when omitted.`,
  ),
  model: z.string().min(1).default(DEFAULT_MODEL).describe(
    "System One model. Pin an exact version for a reproducible study — an " +
      "unpinned model confounds version drift with measurement noise.",
  ),
  baseUrl: z.string().url().default(DEFAULT_BASE_URL),
  timeoutMs: z.number().int().positive().default(30_000),
  maxRetries: z.number().int().min(0).max(10).default(2),
});

type GlobalArgs = z.infer<typeof GlobalArgsSchema>;

function resolveApiKey(ga: GlobalArgs): string {
  const key = ga.apiKey ?? Deno.env.get(API_KEY_ENV);
  if (!key) {
    throw new Error(
      `No TypeSafe API key. Set globalArguments.apiKey or ${API_KEY_ENV}.`,
    );
  }
  // An unevaluated expression is a non-empty string, so a naive presence test
  // passes while the vault is still empty. Reject it explicitly.
  if (key.includes("${{")) {
    throw new Error(
      "TypeSafe API key is still an unresolved expression — the referenced " +
        "vault key is missing. Store it with: swamp vault put <vault> " +
        `${API_KEY_ENV}`,
    );
  }
  return key;
}

// ---------------------------------------------------------------------------
// Framings
//
// P-null: changes that CANNOT change a correct answer. Variance here is
//         unambiguously instrument noise.
// P-para: intended-equivalent rewordings. Variance here is noise + construct
//         drift. Never pooled with P-null; their contrast identifies drift.
//
// NOTE: reversing criteria order is NOT null for a Score question — the
// criteria are ordered levels, so reversing them inverts the scale. Excluded.
// Some transforms are no-ops for some questions (quote_style does nothing to
// instructions containing no apostrophes); the request hash detects that, and
// the report refuses to count a framing whose request is byte-identical to
// another's.
// ---------------------------------------------------------------------------

type FramingType = "null" | "para";

interface Framing {
  id: string;
  type: FramingType;
  state?: (s: Record<string, unknown>) => Record<string, unknown>;
  instructions?: (i: string) => string;
}

const NULL_FRAMINGS: Framing[] = [
  { id: "identity", type: "null" },
  { id: "trailing_ws", type: "null", instructions: (i) => `${i}  ` },
  {
    id: "key_reorder",
    type: "null",
    // Reverse the INSERTION order. Sorting then reversing looked like a
    // reorder but returned the original order for {text, task}, making this
    // framing byte-identical to identity — silently, for the whole study,
    // until the no-op detector caught it.
    state: (s) => {
      const out: Record<string, unknown> = {};
      for (const k of Object.keys(s).reverse()) out[k] = s[k];
      return out;
    },
  },
  {
    id: "irrelevant_field",
    type: "null",
    state: (s) => ({ ...s, record_locator: "XJ-4417" }),
  },
  {
    id: "quote_style",
    type: "null",
    instructions: (i) => i.replace(/'/g, "’"),
  },
];

const DEFAULT_NULL_FRAMINGS = [
  "identity",
  "trailing_ws",
  "key_reorder",
  "irrelevant_field",
];

// ---------------------------------------------------------------------------
// API client
// ---------------------------------------------------------------------------

interface JevQuestion {
  type: "score" | "noul" | "choice";
  instructions: string;
  criteria?: string[] | Record<string, string>;
}

interface JevAnswer {
  type: string;
  /** Score: probability-weighted value across ordered levels. */
  score?: number;
  /** Noul: the probability that the answer is yes. */
  noul?: number;
  /** Choice: the selected option. */
  choice?: string;
  probabilities?: Record<string, number>;
  confidence?: number;
}

interface JevResponse {
  model: string;
  answers: Record<string, JevAnswer>;
}

async function sha256(s: string): Promise<string> {
  const buf = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(s),
  );
  return Array.from(new Uint8Array(buf))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

async function callJev(
  ga: GlobalArgs,
  body: string,
  signal?: AbortSignal,
): Promise<JevResponse> {
  const apiKey = resolveApiKey(ga);
  const baseUrl = (ga.baseUrl ?? DEFAULT_BASE_URL).replace(/\/+$/, "");
  const maxRetries = ga.maxRetries ?? 2;
  const timeoutMs = ga.timeoutMs ?? 30_000;

  for (let attempt = 0;; attempt++) {
    signal?.throwIfAborted();
    const ac = AbortSignal.timeout(timeoutMs);
    const sig = signal ? AbortSignal.any([signal, ac]) : ac;

    let resp: Response;
    try {
      resp = await fetch(`${baseUrl}/v1/systemone`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
          Accept: "application/json",
          "User-Agent": "jev-reliability/2",
        },
        body,
        signal: sig,
      });
    } catch (err) {
      if (signal?.aborted) throw signal.reason;
      if (attempt >= maxRetries) {
        throw new Error(`TypeSafe API unreachable: ${err}`);
      }
      await new Promise((r) => setTimeout(r, 500 * 2 ** attempt));
      continue;
    }

    if (resp.ok) return await resp.json() as JevResponse;

    if (attempt >= maxRetries || !RETRYABLE.has(resp.status)) {
      const t = await resp.text().catch(() => "");
      throw new Error(`TypeSafe API ${resp.status}: ${t.slice(0, 500)}`);
    }
    const ra = resp.headers.get("retry-after");
    await new Promise((r) =>
      setTimeout(
        r,
        ra ? Math.min(Number(ra) * 1000 || 500, 60_000) : 500 * 2 ** attempt,
      )
    );
  }
}

// ---------------------------------------------------------------------------
// Schemas
// ---------------------------------------------------------------------------

const ItemSchema = z.object({
  id: z.string().min(1),
  text: z.string().min(1),
});

const QuestionTypeSchema = z.enum(["score", "noul", "choice"]).default("score")
  .describe(
    "Which System One primitive is under test. score = rating/ranking, " +
      "noul = a yes/no gate, choice = a router.",
  );

const ObservationSchema = z.object({
  item: z.string(),
  framing: z.string(),
  framingType: z.enum(["null", "para"]),
  repeat: z.number().int(),
  requestHash: z.string(),
  questionType: z.string(),
  /** The scalar under analysis: score, P(yes), or P(selected option). */
  value: z.number().nullable(),
  /** The discrete outcome: modal level, yes/no, or the chosen option. */
  decision: z.string().nullable(),
  score: z.number().nullable(),
  noul: z.number().nullable(),
  choice: z.string().nullable(),
  confidence: z.number().nullable(),
  probabilities: z.record(z.string(), z.number()).nullable(),
  probsHash: z.string().nullable(),
  answerable: z.number().nullable(),
  answerableProbs: z.record(z.string(), z.number()).nullable(),
  answerableRaw: z.string().nullable(),
  model: z.string(),
  error: z.string().optional(),
});

type Observation = z.infer<typeof ObservationSchema>;

const ReplicateResultSchema = z.object({
  run: z.string(),
  collectedAt: z.string(),
  model: z.string(),
  questionType: z.string(),
  threshold: z.number(),
  fitIterations: z.number().int(),
  question: z.object({
    instructions: z.string(),
    criteria: z.array(z.string()),
  }),
  design: z.object({
    items: z.number().int(),
    framings: z.array(z.object({ id: z.string(), type: z.string() })),
    repeats: z.number().int(),
    totalCalls: z.number().int(),
  }),
  observations: z.array(ObservationSchema),
});

/** Shared shape for both methods; `check` defaults most of it. */
const GridArgsShape = {
  name: z.string().min(1).describe("Run name; used for the data resource."),
  items: z.array(ItemSchema).min(1).describe(
    "Items to judge — the 'plate' facet. Include items that land mid-scale: " +
      "a question tested only on obvious cases will look perfectly stable.",
  ),
  questionType: QuestionTypeSchema,
  instructions: z.string().min(1).describe("The question under test."),
  criteria: z.array(z.string().min(1)).default([]).describe(
    "Score: ordered levels, lowest first. Choice: the options. Noul: leave " +
      "empty, or use criteriaBool.",
  ),
  criteriaBool: z.object({ true: z.string(), false: z.string() }).optional()
    .describe("Noul only: what yes and no mean."),
  paraphrases: z.array(z.string().min(1)).default([]).describe(
    "Intended-equivalent rewordings of `instructions` — the P-para framings. " +
      "Without these, question-phrasing sensitivity cannot be measured.",
  ),
  nullFramings: z.array(z.string()).default(DEFAULT_NULL_FRAMINGS).describe(
    "P-null framings: identity, trailing_ws, key_reorder, irrelevant_field, " +
      "quote_style.",
  ),
  repeats: z.number().int().min(1).max(50).default(2).describe(
    "Repeat calls per (item, framing) cell — the 'measurement cycle' facet.",
  ),
  threshold: z.number().min(0).max(1).default(0.5).describe(
    "Decision cut-off for a noul question. Also the cut-off the report " +
      "checks for flapping.",
  ),
  fitIterations: z.number().int().min(1000).max(500_000).default(40_000)
    .describe(
      "Sampler iterations the report should use. Raise it if the report says " +
        "the fit did not converge.",
    ),
  answerability: z.string().optional().describe(
    "Optional Noul asked in the SAME request: does this text support " +
      "answering at all? A stability gate detects jitter around a value; it " +
      "cannot detect that no value is warranted.",
  ),
  answerabilityCriteria: z.object({
    true: z.string(),
    false: z.string(),
  }).optional().describe("What yes and no mean for the answerability Noul."),
  dryRun: z.boolean().default(false).describe(
    "Print the plan and call count, then stop without spending anything.",
  ),
};

const GridArgsSchema = z.object(GridArgsShape);
type GridArgs = z.infer<typeof GridArgsSchema>;

// ---------------------------------------------------------------------------
// Answer interpretation — one shape per primitive
// ---------------------------------------------------------------------------

function argmax(p: Record<string, number> | undefined): string | null {
  if (!p) return null;
  let best: string | null = null, bv = -Infinity;
  for (const [k, v] of Object.entries(p)) {
    if (v > bv) { bv = v; best = k; }
  }
  return best;
}

interface Interpreted {
  value: number | null;
  decision: string | null;
}

function interpret(
  a: JevAnswer,
  type: GridArgs["questionType"],
  threshold: number,
): Interpreted {
  if (type === "noul") {
    const v = typeof a.noul === "number" ? a.noul : null;
    return { value: v, decision: v === null ? null : v >= threshold ? "yes" : "no" };
  }
  if (type === "choice") {
    const sel = a.choice ?? argmax(a.probabilities);
    const v = sel && a.probabilities ? a.probabilities[sel] ?? null : null;
    return { value: v, decision: sel };
  }
  return { value: a.score ?? null, decision: argmax(a.probabilities) };
}

function extractNoul(a: JevAnswer): number | null {
  if (typeof a.noul === "number") return a.noul;
  if (typeof a.score === "number") return a.score;
  const p = a.probabilities;
  if (!p) return null;
  for (const k of ["true", "yes", "1", "TRUE", "Yes"]) {
    if (typeof p[k] === "number") return p[k];
  }
  return null;
}

/** Stable stringify so identical distributions hash identically. */
function canonical(o: Record<string, number>): string {
  return JSON.stringify(Object.keys(o).sort().map((k) => [k, o[k]]));
}

// ---------------------------------------------------------------------------
// The grid runner, shared by both methods
// ---------------------------------------------------------------------------

// deno-lint-ignore no-explicit-any
async function runGrid(args: GridArgs, context: any) {
  const ga = context.globalArgs as GlobalArgs;
  const model = ga.model ?? DEFAULT_MODEL;

  if (args.questionType !== "noul" && args.criteria.length < 2) {
    throw new Error(
      `A ${args.questionType} question needs at least 2 criteria ` +
        `(${args.questionType === "score" ? "ordered levels" : "options"}).`,
    );
  }

  const framings: Framing[] = [
    ...NULL_FRAMINGS.filter((f) => args.nullFramings.includes(f.id)),
    ...args.paraphrases.map((p, i) => ({
      id: `para_${i + 1}`,
      type: "para" as const,
      instructions: () => p,
    })),
  ];
  if (framings.length === 0) {
    throw new Error("No framings selected — nothing to measure.");
  }

  const total = args.items.length * framings.length * args.repeats;
  context.logger.info(
    "Plan: {items} items x {framings} framings ({nulls} null, {paras} " +
      "paraphrase) x {repeats} repeats = {total} calls against {model}",
    {
      items: args.items.length,
      framings: framings.length,
      nulls: framings.filter((f) => f.type === "null").length,
      paras: framings.filter((f) => f.type === "para").length,
      repeats: args.repeats,
      total,
      model,
    },
  );
  if (!args.paraphrases.length) {
    context.logger.info(
      "No paraphrases given — question-phrasing sensitivity will NOT be " +
        "measured, and that is usually the largest term.",
    );
  }
  if (args.dryRun) {
    context.logger.info("dryRun set — stopping before any API call.");
    return { dataHandles: [] };
  }

  const observations: Observation[] = [];
  // framing id -> request hash, per item; used to catch no-op framings.
  const hashByItem = new Map<string, Map<string, string>>();

  for (const item of args.items) {
    for (const framing of framings) {
      const baseState: Record<string, unknown> = {
        text: item.text,
        task: "Judge this text against the question below.",
      };
      const state = framing.state ? framing.state(baseState) : baseState;
      const instructions = framing.instructions
        ? framing.instructions(args.instructions)
        : args.instructions;

      const q: JevQuestion = { type: args.questionType, instructions };
      if (args.questionType === "noul") {
        if (args.criteriaBool) q.criteria = args.criteriaBool;
      } else {
        q.criteria = args.criteria;
      }
      const questions: Record<string, JevQuestion> = { q };
      if (args.answerability) {
        questions.a = {
          type: "noul",
          instructions: args.answerability,
          ...(args.answerabilityCriteria
            ? { criteria: args.answerabilityCriteria }
            : {}),
        };
      }

      const body = JSON.stringify({ state, model, questions });
      const requestHash = await sha256(body);
      if (!hashByItem.has(item.id)) hashByItem.set(item.id, new Map());
      hashByItem.get(item.id)!.set(framing.id, requestHash);

      for (let r = 1; r <= args.repeats; r++) {
        context.signal?.throwIfAborted?.();
        const base = {
          item: item.id,
          framing: framing.id,
          framingType: framing.type,
          repeat: r,
          requestHash,
          questionType: args.questionType,
        };
        try {
          const resp = await callJev(ga, body, context.signal);
          const a = resp.answers.q ?? {};
          const probs = a.probabilities ?? null;
          const ans = resp.answers.a;
          const { value, decision } = interpret(
            a,
            args.questionType,
            args.threshold,
          );
          observations.push({
            ...base,
            value,
            decision,
            score: a.score ?? null,
            noul: a.noul ?? null,
            choice: a.choice ?? null,
            confidence: a.confidence ?? null,
            probabilities: probs,
            probsHash: probs ? await sha256(canonical(probs)) : null,
            answerable: ans ? extractNoul(ans) : null,
            answerableProbs: ans?.probabilities ?? null,
            answerableRaw: ans ? JSON.stringify(ans) : null,
            model: resp.model,
          });
        } catch (err) {
          observations.push({
            ...base,
            value: null,
            decision: null,
            score: null,
            noul: null,
            choice: null,
            confidence: null,
            probabilities: null,
            probsHash: null,
            answerable: null,
            answerableProbs: null,
            answerableRaw: null,
            model,
            error: (err as Error).message,
          });
        }
      }
    }
  }

  // No-op framings: a "perturbation" that produced a byte-identical request
  // measured nothing and would otherwise pad the framing count.
  const noop = new Set<string>();
  for (const perItem of hashByItem.values()) {
    const seen = new Map<string, string>();
    for (const [fid, h] of perItem) {
      if (seen.has(h)) noop.add(`${fid} == ${seen.get(h)}`);
      else seen.set(h, fid);
    }
  }
  if (noop.size) {
    context.logger.info(
      "No-op framings detected (byte-identical requests): {pairs}",
      { pairs: [...noop].join(", ") },
    );
  }

  const ok = observations.filter((o) => !o.error).length;
  context.logger.info("Collected {ok}/{total} observations ({fail} failed)", {
    ok,
    total,
    fail: total - ok,
  });

  const handle = await context.writeResource(
    "replicate",
    `replicate-${args.name}`,
    {
      run: args.name,
      collectedAt: new Date().toISOString(),
      model,
      questionType: args.questionType,
      threshold: args.threshold,
      fitIterations: args.fitIterations,
      question: {
        instructions: args.instructions,
        criteria: args.criteria,
      },
      design: {
        items: args.items.length,
        framings: framings.map((f) => ({ id: f.id, type: f.type })),
        repeats: args.repeats,
        totalCalls: total,
      },
      observations,
    },
  );
  return { dataHandles: [handle] };
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

export const model = {
  type: "@vcjdeboer/jev-reliability",
  version: "2026.09.19.2",
  globalArguments: GlobalArgsSchema,
  upgrades: [],
  reports: ["@vcjdeboer/jev-reliability-report"],

  resources: {
    replicate: {
      description:
        "Raw nested replicate grid — one row per (item, framing, repeat).",
      schema: ReplicateResultSchema,
      lifetime: "infinite",
      garbageCollection: 100,
    },
  },

  checks: {
    "api-key-configured": {
      description: "Ensure a TypeSafe API key is available",
      labels: ["policy"],
      execute: async (context: { globalArgs: GlobalArgs }) => {
        try {
          resolveApiKey(context.globalArgs);
          return await Promise.resolve({ pass: true });
        } catch (err) {
          return { pass: false, errors: [(err as Error).message] };
        }
      },
    },
  },

  methods: {
    check: {
      description:
        "Is this question safe to build on? Give a question and some items; " +
        "the design is defaulted. Prints the call count first — use dryRun " +
        "to see the cost without spending it. Then read the report.",
      arguments: GridArgsSchema,
      // deno-lint-ignore no-explicit-any
      execute: (args: GridArgs, context: any) => runGrid(args, context),
    },

    replicate: {
      description:
        "Low-level escape hatch: run an explicit item x framing x repeat " +
        "grid. Prefer `check` unless you are designing the study yourself.",
      arguments: GridArgsSchema,
      // deno-lint-ignore no-explicit-any
      execute: (args: GridArgs, context: any) => runGrid(args, context),
    },
  },
};
