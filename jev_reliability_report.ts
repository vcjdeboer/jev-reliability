/**
 * @vcjdeboer/jev-reliability-report — pre-flight verdict for a Jev question.
 *
 * Consumes the raw replicate grids recorded by @vcjdeboer/jev-reliability and
 * answers the questions you need settled BEFORE building on a Jev question:
 *
 *   1. Repeatability   — does an identical request give an identical answer?
 *   2. Framing         — does rewording the question move the answer, and by
 *                        how much more than plain noise?
 *   3. Resolution      — can this question actually tell your items apart, or
 *                        does the scale saturate into ties?
 *   4. Answerability   — is it confidently rating things that have nothing to
 *                        rate?
 *
 * The variance decomposition is a nested Bayesian measurement model
 * (repeat -> framing -> item, after OCRbayes; Zhang, Yuan, Keijer & de Boer,
 * PLOS ONE 2021), fitted by a Gibbs/slice sampler implemented here in
 * TypeScript so the report needs no Stan, R or Python.
 *
 * It refuses to print posterior numbers when the convergence gate fails.
 *
 * @module
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface DataEntry {
  name: string;
  version?: number;
  tags?: Record<string, string>;
}

interface Observation {
  item: string;
  framing: string;
  framingType: "null" | "para";
  repeat: number;
  requestHash: string;
  questionType?: string;
  /** Generalised scalar; older grids carry only `score`. */
  value?: number | null;
  /** Generalised discrete outcome; older grids derive it from the argmax. */
  decision?: string | null;
  score: number | null;
  noul?: number | null;
  choice?: string | null;
  confidence: number | null;
  probabilities: Record<string, number> | null;
  probsHash: string | null;
  answerable?: number | null;
  model: string;
  error?: string;
}

interface Grid {
  run: string;
  collectedAt: string;
  model: string;
  questionType?: string;
  threshold?: number;
  fitIterations?: number;
  question: { instructions: string; criteria: string[] };
  design: {
    items: number;
    framings: { id: string; type: string }[];
    repeats: number;
    totalCalls: number;
  };
  observations: Observation[];
}

/**
 * The scalar under analysis. Grids written before the primitive
 * generalisation carry only `score`, so fall back to it.
 */
const val = (o: Observation): number | null =>
  o.value !== undefined && o.value !== null ? o.value : o.score;

function argmaxKey(p: Record<string, number> | null): string | null {
  if (!p) return null;
  let best: string | null = null, bv = -Infinity;
  for (const [k, v] of Object.entries(p)) {
    if (v > bv) { bv = v; best = k; }
  }
  return best;
}

/** The discrete outcome — what the calling code would actually act on. */
const decisionOf = (o: Observation): string | null =>
  o.decision !== undefined && o.decision !== null
    ? o.decision
    : argmaxKey(o.probabilities);

// ---------------------------------------------------------------------------
// Small stats helpers
// ---------------------------------------------------------------------------

const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;

function pvar(v: number[]): number {
  if (v.length < 2) return 0;
  const m = mean(v);
  return v.reduce((a, x) => a + (x - m) ** 2, 0) / v.length;
}

function svar(v: number[]): number {
  if (v.length < 2) return 0;
  const m = mean(v);
  return v.reduce((a, x) => a + (x - m) ** 2, 0) / (v.length - 1);
}

const quantile = (sorted: number[], q: number) =>
  sorted[Math.min(sorted.length - 1, Math.max(0, Math.floor(q * sorted.length)))];

/** Deterministic PRNG — a report must produce the same verdict every run. */
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeGauss(rand: () => number) {
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const s = spare;
      spare = null;
      return s;
    }
    let u = 0, v = 0, s = 0;
    do {
      u = rand() * 2 - 1;
      v = rand() * 2 - 1;
      s = u * u + v * v;
    } while (s === 0 || s >= 1);
    const f = Math.sqrt((-2 * Math.log(s)) / s);
    spare = v * f;
    return u * f;
  };
}

// ---------------------------------------------------------------------------
// Nested measurement model — Gibbs with slice-sampled scales
// ---------------------------------------------------------------------------

interface Draw {
  mu: number;
  sItem: number;
  sFram: number;
  sNull: number;
  sPara: number;
  sigmaY: number;
  drift: number;
  G: number;
}

function scaleLogPost(vals: number[], s: number, priorScale: number): number {
  if (s <= 0) return -Infinity;
  const ss = vals.reduce((a, x) => a + x * x, 0);
  return -0.5 * (s * s) / (priorScale * priorScale) -
    vals.length * Math.log(s) - 0.5 * ss / (s * s);
}

function sliceSample(
  x0: number,
  logp: (x: number) => number,
  rand: () => number,
  w = 0.05,
  lower = 1e-9,
  upper = 10,
): number {
  const y = logp(x0) + Math.log(rand());
  let lo = Math.max(lower, x0 - w * rand());
  let hi = Math.min(upper, lo + w);
  for (let i = 0; i < 50 && lo > lower && logp(lo) > y; i++) {
    lo = Math.max(lower, lo - w);
  }
  for (let i = 0; i < 50 && hi < upper && logp(hi) > y; i++) {
    hi = Math.min(upper, hi + w);
  }
  for (let i = 0; i < 100; i++) {
    const x = lo + rand() * (hi - lo);
    if (logp(x) > y) return x;
    if (x < x0) lo = x;
    else hi = x;
  }
  return x0;
}

function runChain(
  y: number[],
  item: number[],
  fram: number[],
  isPara: boolean[],
  nItems: number,
  nFram: number,
  priorScale: number,
  iters: number,
  burn: number,
  seed: number,
): Draw[] {
  const rand = mulberry32(seed);
  const gauss = makeGauss(rand);
  const N = y.length;

  const cellKey = (i: number, j: number) => i * nFram + j;
  const cellObs = new Map<number, number[]>();
  const itemObs: number[][] = Array.from({ length: nItems }, () => []);
  const framObs: number[][] = Array.from({ length: nFram }, () => []);
  for (let n = 0; n < N; n++) {
    const k = cellKey(item[n], fram[n]);
    if (!cellObs.has(k)) cellObs.set(k, []);
    cellObs.get(k)!.push(n);
    itemObs[item[n]].push(n);
    framObs[fram[n]].push(n);
  }

  let mu = mean(y);
  const a = new Array(nItems).fill(0);
  const f = new Array(nFram).fill(0);
  const g = new Map<number, number>();
  for (const k of cellObs.keys()) g.set(k, 0);
  let sItem = 0.1, sFram = 0.1, sNull = 0.1, sPara = 0.1, sigmaY = 0.1;

  const draws: Draw[] = [];
  for (let it = 0; it < iters; it++) {
    const sy2 = sigmaY * sigmaY;

    // mu
    let prec = 1 + N / sy2;
    let acc = 1.5;
    for (let n = 0; n < N; n++) {
      acc += (y[n] - a[item[n]] - f[fram[n]] -
        g.get(cellKey(item[n], fram[n]))!) / sy2;
    }
    mu = acc / prec + gauss() / Math.sqrt(prec);

    // item effects
    for (let i = 0; i < nItems; i++) {
      const idx = itemObs[i];
      prec = 1 / (sItem * sItem) + idx.length / sy2;
      acc = 0;
      for (const n of idx) {
        acc += (y[n] - mu - f[fram[n]] - g.get(cellKey(i, fram[n]))!) / sy2;
      }
      a[i] = acc / prec + gauss() / Math.sqrt(prec);
    }

    // framing main effects
    for (let j = 0; j < nFram; j++) {
      const idx = framObs[j];
      prec = 1 / (sFram * sFram) + idx.length / sy2;
      acc = 0;
      for (const n of idx) {
        acc += (y[n] - mu - a[item[n]] - g.get(cellKey(item[n], j))!) / sy2;
      }
      f[j] = acc / prec + gauss() / Math.sqrt(prec);
    }

    // item x framing interaction
    for (const [k, idx] of cellObs) {
      const i = Math.floor(k / nFram), j = k % nFram;
      const sg = isPara[j] ? sPara : sNull;
      prec = 1 / (sg * sg) + idx.length / sy2;
      acc = 0;
      for (const n of idx) acc += (y[n] - mu - a[i] - f[j]) / sy2;
      g.set(k, acc / prec + gauss() / Math.sqrt(prec));
    }

    // Ridge-breaking joint shifts. (mu + d, a - d) has an identical
    // likelihood, so plain Gibbs crawls along that ridge and mu fails to mix.
    // Symmetric proposal, likelihood invariant -> prior ratio only.
    for (const [vec, sVec] of [[a, sItem], [f, sFram]] as [number[], number][]) {
      const d = gauss() * 0.05;
      let logr = -0.5 * ((mu + d - 1.5) ** 2 - (mu - 1.5) ** 2);
      for (const v of vec) {
        logr += -0.5 * ((v - d) ** 2 - v * v) / (sVec * sVec);
      }
      if (Math.log(rand()) < logr) {
        mu += d;
        for (let k = 0; k < vec.length; k++) vec[k] -= d;
      }
    }

    // scales
    sItem = sliceSample(sItem, (s) => scaleLogPost(a, s, priorScale), rand);
    sFram = sliceSample(sFram, (s) => scaleLogPost(f, s, priorScale), rand);
    const gn: number[] = [], gp: number[] = [];
    for (const [k, v] of g) (isPara[k % nFram] ? gp : gn).push(v);
    sNull = sliceSample(sNull, (s) => scaleLogPost(gn, s, priorScale), rand);
    sPara = sliceSample(sPara, (s) => scaleLogPost(gp, s, priorScale), rand);
    const resid = y.map((v, n) =>
      v - mu - a[item[n]] - f[fram[n]] - g.get(cellKey(item[n], fram[n]))!
    );
    sigmaY = sliceSample(
      sigmaY,
      (s) => scaleLogPost(resid, s, priorScale),
      rand,
    );

    if (it >= burn) {
      draws.push({
        mu,
        sItem,
        sFram,
        sNull,
        sPara,
        sigmaY,
        drift: sPara * sPara - sNull * sNull,
        G: (sItem * sItem) /
          (sItem * sItem + sNull * sNull + sigmaY * sigmaY),
      });
    }
  }
  return draws;
}

function splitRhat(chains: number[][]): number {
  const halves: number[][] = [];
  for (const c of chains) {
    const h = Math.floor(c.length / 2);
    halves.push(c.slice(0, h), c.slice(h, 2 * h));
  }
  const m = halves.length, n = halves[0].length;
  const means = halves.map(mean);
  const grand = mean(means);
  const B = (n / (m - 1)) * means.reduce((a, x) => a + (x - grand) ** 2, 0);
  const W = mean(halves.map((h, k) =>
    h.reduce((a, x) => a + (x - means[k]) ** 2, 0) / (n - 1)
  ));
  if (W <= 0) return NaN;
  return Math.sqrt((((n - 1) / n) * W + B / n) / W);
}

function ess(chains: number[][]): number {
  const m = chains.length, n = chains[0].length;
  const means = chains.map(mean);
  const grand = mean(means);
  const W = mean(chains.map((c, k) =>
    c.reduce((a, x) => a + (x - means[k]) ** 2, 0) / (n - 1)
  ));
  if (W <= 0) return NaN;
  const acov = (lag: number) =>
    mean(chains.map((c, k) => {
      let t = 0;
      for (let i = 0; i < n - lag; i++) {
        t += (c[i] - means[k]) * (c[i + lag] - means[k]);
      }
      return t / n;
    }));
  const var0 = acov(0);
  if (var0 <= 0) return NaN;
  let tau = 1, t = 1;
  while (t < n - 2) {
    const p = acov(t) / var0 + acov(t + 1) / var0;
    if (p <= 0) break;
    tau += 2 * p;
    t += 2;
  }
  const B = m > 1
    ? (n / (m - 1)) * means.reduce((a, x) => a + (x - grand) ** 2, 0)
    : 0;
  return (m * n * ((((n - 1) / n) * W + B / n) / var0)) / Math.max(tau, 1e-9);
}

interface FitSummary {
  param: string;
  mean: number;
  lo: number;
  hi: number;
  rhat: number;
  ess: number;
}

interface FitResult {
  rows: FitSummary[];
  maxRhat: number;
  minEss: number;
  converged: boolean;
  iters: number;
}

const PARAMS: (keyof Draw)[] = [
  "mu",
  "sItem",
  "sFram",
  "sNull",
  "sPara",
  "sigmaY",
  "drift",
  "G",
];

function fitModel(
  y: number[],
  item: number[],
  fram: number[],
  isPara: boolean[],
  nItems: number,
  nFram: number,
  iters: number,
): FitResult {
  const burn = Math.floor(iters / 4);
  const chains = [0, 1, 2, 3].map((k) =>
    runChain(y, item, fram, isPara, nItems, nFram, 1, iters, burn, 1000 + k)
  );
  const rows: FitSummary[] = [];
  let maxRhat = 0, minEss = Infinity;
  for (const p of PARAMS) {
    const cs = chains.map((c) => c.map((d) => d[p]));
    const flat = cs.flat().slice().sort((x, z) => x - z);
    const r = splitRhat(cs), e = ess(cs);
    if (!Number.isNaN(r)) maxRhat = Math.max(maxRhat, r);
    if (!Number.isNaN(e)) minEss = Math.min(minEss, e);
    rows.push({
      param: p,
      mean: mean(flat),
      lo: quantile(flat, 0.025),
      hi: quantile(flat, 0.975),
      rhat: r,
      ess: e,
    });
  }
  return {
    rows,
    maxRhat,
    minEss,
    converged: maxRhat < 1.01 && minEss > 400,
    iters,
  };
}

// ---------------------------------------------------------------------------
// Analysis of one grid
// ---------------------------------------------------------------------------

interface Analysis {
  run: string;
  model: string;
  collectedAt: string;
  design: Grid["design"];
  ok: number;
  failed: number;
  repeatSd: number | null;
  deterministicItems: number;
  jitterItems: number;
  nullSd: number | null;
  paraSd: number | null;
  driftRatio: number | null;
  itemMeans: Record<string, number>;
  distinctMeans: number;
  exactTies: { value: number; items: string[] }[];
  resolution: number | null;
  indistinguishablePairs: number;
  totalPairs: number;
  modalMoves: number;
  thresholdFlappers: { item: string; lo: number; hi: number }[];
  answerability: {
    n: number;
    lowAnswerHighConf: { item: string; conf: number; ans: number }[];
    min: number;
    max: number;
  } | null;
  fit: FitResult | null;
  saturatedExcluded: number;
  questionType: string;
  /** Fraction of calls whose decision differs from their item's modal one. */
  flipAll: number;
  /** Same, restricted to repeats and semantically-null perturbations. */
  flipStable: number;
  /** Same, restricted to paraphrase framings. */
  flipPara: number | null;
  flippedItems: string[];
  noopFramings: string[];
  warnings: string[];
}

function distinctFraction(m: Record<string, number>): number {
  const n = Object.keys(m).length;
  if (!n) return 1;
  return new Set(Object.values(m).map((x) => Math.round(x * 1e6))).size / n;
}

function analyse(grid: Grid, iters: number): Analysis {
  const all = grid.observations;
  const obs = all.filter((o) => val(o) !== null && !o.error);
  const failed = all.length - obs.length;

  // --- cells -------------------------------------------------------------
  const cells = new Map<string, Observation[]>();
  for (const o of obs) {
    const k = `${o.item}|${o.framing}`;
    if (!cells.has(k)) cells.set(k, []);
    cells.get(k)!.push(o);
  }

  // --- repeatability -----------------------------------------------------
  const withinVars: number[] = [];
  for (const v of cells.values()) {
    if (v.length > 1) withinVars.push(pvar(v.map((o) => val(o)!)));
  }
  const repeatSd = withinVars.length ? Math.sqrt(mean(withinVars)) : null;

  const byItem = new Map<string, Observation[]>();
  for (const o of obs) {
    if (!byItem.has(o.item)) byItem.set(o.item, []);
    byItem.get(o.item)!.push(o);
  }

  let deterministicItems = 0, jitterItems = 0;
  for (const v of byItem.values()) {
    const ident = v.filter((o) => o.framing === "identity");
    if (ident.length < 2) continue;
    const hashes = new Set(ident.map((o) => o.probsHash));
    if (hashes.size === 1) deterministicItems++;
    else jitterItems++;
  }

  // --- framing sensitivity ------------------------------------------------
  const cellMeans = new Map<string, { null: number[]; para: number[] }>();
  for (const [k, v] of cells) {
    const item = k.split("|")[0];
    if (!cellMeans.has(item)) cellMeans.set(item, { null: [], para: [] });
    cellMeans.get(item)![v[0].framingType].push(mean(v.map((o) => val(o)!)));
  }
  const nRep = grid.design.repeats;
  const debias = (vals: number[][]) => {
    const parts = vals.filter((m) => m.length > 1).map(pvar);
    if (!parts.length) return null;
    const raw = mean(parts);
    const corrected = repeatSd !== null
      ? Math.max(raw - (repeatSd * repeatSd) / nRep, 0)
      : raw;
    return Math.sqrt(corrected);
  };
  const nullSd = debias([...cellMeans.values()].map((x) => x.null));
  const paraSd = debias([...cellMeans.values()].map((x) => x.para));
  const driftRatio = nullSd && paraSd && nullSd > 0 ? paraSd / nullSd : null;

  // --- resolution ---------------------------------------------------------
  const itemMeans: Record<string, number> = {};
  for (const [k, v] of byItem) itemMeans[k] = mean(v.map((o) => val(o)!));
  const byValue = new Map<number, string[]>();
  for (const [k, m] of Object.entries(itemMeans)) {
    const r = Math.round(m * 1e6) / 1e6;
    if (!byValue.has(r)) byValue.set(r, []);
    byValue.get(r)!.push(k);
  }
  const exactTies = [...byValue.entries()]
    .filter(([, ids]) => ids.length > 1)
    .map(([value, items]) => ({ value, items: items.sort() }))
    .sort((x, z) => x.value - z.value);

  const noiseSd = paraSd ?? repeatSd;
  const resolution = noiseSd !== null ? 1.96 * Math.SQRT2 * noiseSd : null;
  const ids = Object.keys(itemMeans).sort();
  let indist = 0, totalPairs = 0;
  for (let i = 0; i < ids.length; i++) {
    for (let j = i + 1; j < ids.length; j++) {
      totalPairs++;
      if (
        resolution !== null &&
        Math.abs(itemMeans[ids[i]] - itemMeans[ids[j]]) < resolution
      ) indist++;
    }
  }

  // --- modal stability + threshold flapping -------------------------------
  let modalMoves = 0;
  const thresholdFlappers: { item: string; lo: number; hi: number }[] = [];
  for (const [k, v] of byItem) {
    const modes = new Set<string>();
    for (const o of v) {
      if (!o.probabilities) continue;
      let best = "", bv = -1;
      for (const [lvl, p] of Object.entries(o.probabilities)) {
        if (p > bv) { bv = p; best = lvl; }
      }
      modes.add(best);
    }
    if (modes.size > 1) modalMoves++;
    const confs = v.map((o) => o.confidence).filter((c): c is number =>
      c !== null
    );
    if (confs.length > 1) {
      const lo = Math.min(...confs), hi = Math.max(...confs);
      for (const t of [0.5, 0.7, 0.8, 0.9]) {
        if (lo < t && hi >= t) {
          thresholdFlappers.push({ item: k, lo, hi });
          break;
        }
      }
    }
  }

  // --- answerability ------------------------------------------------------
  const withAns = obs.filter((o) =>
    o.answerable !== null && o.answerable !== undefined
  );
  let answerability: Analysis["answerability"] = null;
  if (withAns.length) {
    const perItem = new Map<string, { a: number[]; c: number[] }>();
    for (const o of withAns) {
      if (!perItem.has(o.item)) perItem.set(o.item, { a: [], c: [] });
      perItem.get(o.item)!.a.push(o.answerable!);
      if (o.confidence !== null) perItem.get(o.item)!.c.push(o.confidence);
    }
    const flags: { item: string; conf: number; ans: number }[] = [];
    const means_: number[] = [];
    for (const [k, v] of perItem) {
      const am = mean(v.a);
      means_.push(am);
      const cm = v.c.length ? mean(v.c) : 0;
      if (am <= 0.5 && cm >= 0.8) flags.push({ item: k, conf: cm, ans: am });
    }
    answerability = {
      n: perItem.size,
      lowAnswerHighConf: flags.sort((x, z) => x.ans - z.ans),
      min: Math.min(...means_),
      max: Math.max(...means_),
    };
  }

  // --- Bayesian fit -------------------------------------------------------
  const framTypes = new Set(obs.map((o) => o.framingType));
  let fit: FitResult | null = null;
  let saturatedExcluded = 0;
  if (framTypes.has("null") && framTypes.has("para") && nRep > 1) {
    // Saturated items are pinned at a scale end: they carry no information
    // about variance components and break a Normal likelihood. Detect them by
    // position on the scale rather than by exactly-zero variance — an item at
    // 0.0001 of the boundary is just as saturated and just as uninformative,
    // and including it inflates the between-item term (and therefore G).
    const top = Math.max(1, (grid.question?.criteria?.length ?? 0) - 1);
    const EDGE = 0.01;
    const sat = new Set<string>();
    for (const [k, v] of byItem) {
      const m = mean(v.map((o) => val(o)!));
      if (m <= EDGE || m >= top - EDGE) sat.add(k);
    }
    saturatedExcluded = sat.size;
    const use = obs.filter((o) => !sat.has(o.item));
    const itemIdx = [...new Set(use.map((o) => o.item))].sort();
    const framIdx = [...new Set(use.map((o) => o.framing))].sort();
    if (itemIdx.length >= 3 && framIdx.length >= 2) {
      const isPara = framIdx.map((fr) =>
        use.find((o) => o.framing === fr)!.framingType === "para"
      );
      fit = fitModel(
        use.map((o) => val(o)!),
        use.map((o) => itemIdx.indexOf(o.item)),
        use.map((o) => framIdx.indexOf(o.framing)),
        isPara,
        itemIdx.length,
        framIdx.length,
        iters,
      );
    }
  }

  // --- decision flip rate -------------------------------------------------
  // The number a non-statistician can act on: if you reran this, how often
  // would the decision your code acts on come out differently?
  // The reference decision is the item's modal decision across ALL of its
  // observations, not the pool's own mode. Using a pool-local mode makes every
  // subset look internally consistent and the rates stop nesting — a subset
  // could read 0% while the overall rate is non-zero, which is unreadable.
  const modalByItem = new Map<string, string | null>();
  for (const [k, v] of byItem) {
    const tally = new Map<string, number>();
    for (const o of v) {
      const d = decisionOf(o);
      if (d === null) continue;
      tally.set(d, (tally.get(d) ?? 0) + 1);
    }
    let modal: string | null = null, best = -1;
    for (const [d, c] of tally) if (c > best) { best = c; modal = d; }
    modalByItem.set(k, modal);
  }
  const flipCount = (pool: Observation[]) => {
    let n = 0, flips = 0;
    for (const o of pool) {
      const d = decisionOf(o);
      if (d === null) continue;
      n++;
      if (d !== modalByItem.get(o.item)) flips++;
    }
    return n ? flips / n : 0;
  };
  const flippedItems: string[] = [];
  for (const [k, v] of byItem) {
    const ds = new Set(v.map(decisionOf).filter((d) => d !== null));
    if (ds.size > 1) flippedItems.push(k);
  }
  const nullPool = obs.filter((o) => o.framingType === "null");
  const paraPool = obs.filter((o) => o.framingType === "para");
  const flipAll = flipCount(obs);
  const flipStable = flipCount(nullPool);
  const flipPara = paraPool.length ? flipCount(paraPool) : null;

  // --- no-op framings ------------------------------------------------------
  // A "perturbation" that produced a byte-identical request measured nothing.
  // Detected from the recorded request hashes, so it works on old grids too.
  const noopSet = new Set<string>();
  const hashesByItem = new Map<string, Map<string, string>>();
  for (const o of obs) {
    if (!hashesByItem.has(o.item)) hashesByItem.set(o.item, new Map());
    hashesByItem.get(o.item)!.set(o.framing, o.requestHash);
  }
  for (const perItem of hashesByItem.values()) {
    const seen = new Map<string, string>();
    for (const [fid, h] of perItem) {
      if (seen.has(h) && seen.get(h) !== fid) {
        noopSet.add(`${fid} == ${seen.get(h)}`);
      } else if (!seen.has(h)) seen.set(h, fid);
    }
  }
  const noopFramings = [...noopSet].sort();

  // --- warnings -----------------------------------------------------------
  const warnings: string[] = [];
  const tiedItems = exactTies.reduce((a, t) => a + t.items.length, 0);
  if (tiedItems) {
    warnings.push(
      `${tiedItems} of ${ids.length} items are EXACT ties — they cannot be ` +
        `ranked against each other at all. Any ordering shown among them is ` +
        `arbitrary.`,
    );
  }
  if (driftRatio && driftRatio > 1.5) {
    warnings.push(
      `Rewording the question moves answers ${driftRatio.toFixed(1)}x more ` +
        `than semantically-null changes do. The exact phrasing is part of ` +
        `your measurement — version it.`,
    );
  }
  if (resolution !== null && totalPairs && indist / totalPairs > 0.1) {
    warnings.push(
      `${indist}/${totalPairs} item pairs (${
        (100 * indist / totalPairs).toFixed(1)
      }%) are closer together than the noise. Ranking them is coin-flipping.`,
    );
  }
  if (thresholdFlappers.length) {
    warnings.push(
      `${thresholdFlappers.length} item(s) have confidence straddling a common ` +
        `cut-off on IDENTICAL input — a fixed threshold will flip run to run.`,
    );
  }
  if (answerability?.lowAnswerHighConf.length) {
    warnings.push(
      `${answerability.lowAnswerHighConf.length} item(s) are rated CONFIDENTLY ` +
        `but score low on answerability — the model is confidently judging ` +
        `something with nothing to judge.`,
    );
  }
  if (!answerability) {
    warnings.push(
      `No answerability question was asked. Confidence cannot detect "there ` +
        `is nothing here to judge" — ask a separate Noul for it.`,
    );
  }
  if (jitterItems && !deterministicItems) {
    warnings.push(
      `Identical requests returned different distributions for ` +
        `${jitterItems} item(s). Do not compare raw numbers for equality.`,
    );
  }
  if (noopFramings.length) {
    warnings.push(
      `${noopFramings.length} framing(s) produced a BYTE-IDENTICAL request to ` +
        `another and therefore measured nothing: ${noopFramings.join(", ")}. ` +
        `They pad the framing count without adding information.`,
    );
  }
  const nItemsTotal = Object.keys(itemMeans).length;
  if (nItemsTotal && distinctFraction(itemMeans) < 0.5) {
    warnings.push(
      `Your item set is degenerate for this question: only ` +
        `${new Set(Object.values(itemMeans).map((m) => Math.round(m * 1e6)))
          .size} distinct values across ${nItemsTotal} items. The verdict ` +
        `below is weak evidence — add items that land mid-scale.`,
    );
  }
  if (flipAll > 0) {
    warnings.push(
      `Decision flip rate ${(100 * flipAll).toFixed(1)}% — that share of ` +
        `calls disagrees with its own item's most common decision.`,
    );
  }
  if (fit && !fit.converged) {
    warnings.push(
      `The Bayesian fit did NOT converge (max Rhat ` +
        `${fit.maxRhat.toFixed(4)}, min ESS ${fit.minEss.toFixed(0)}). ` +
        `Posterior numbers are withheld. Re-run with more iterations.`,
    );
  }

  return {
    run: grid.run,
    model: grid.model,
    collectedAt: grid.collectedAt,
    design: grid.design,
    ok: obs.length,
    failed,
    repeatSd,
    deterministicItems,
    jitterItems,
    nullSd,
    paraSd,
    driftRatio,
    itemMeans,
    distinctMeans: byValue.size,
    exactTies,
    resolution,
    indistinguishablePairs: indist,
    totalPairs,
    modalMoves,
    thresholdFlappers,
    answerability,
    fit,
    saturatedExcluded,
    questionType: grid.questionType ?? "score",
    flipAll,
    flipStable,
    flipPara,
    flippedItems,
    noopFramings,
    warnings,
  };
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

const f4 = (x: number | null) => x === null ? "—" : x.toFixed(4);

const LABEL: Record<string, string> = {
  mu: "grand mean",
  sItem: "between items",
  sFram: "framing main effect",
  sNull: "item x null-framing",
  sPara: "item x paraphrase",
  sigmaY: "repeat (residual)",
  drift: "construct drift (variance)",
  G: "reliability G",
};

function renderOne(a: Analysis): string {
  const L: string[] = [];
  L.push(`## ${a.run}`);
  L.push("");
  L.push(
    `\`${a.model}\` · ${a.questionType} question · ${a.collectedAt} · ` +
      `${a.design.items} items × ` +
      `${a.design.framings.length} framings × ${a.design.repeats} repeats = ` +
      `${a.design.totalCalls} calls (${a.ok} ok, ${a.failed} failed)`,
  );
  L.push("");

  const pct = (x: number) => `${(100 * x).toFixed(1)}%`;
  L.push("### Headline — decision flip rate");
  L.push("");
  L.push(
    "If you reran this question, how often would the decision your code " +
      "acts on come out differently?",
  );
  L.push("");
  L.push("| condition | flip rate |");
  L.push("|---|---|");
  L.push(`| same request, repeated | **${pct(a.flipStable)}** |`);
  if (a.flipPara !== null) {
    L.push(`| question reworded | **${pct(a.flipPara)}** |`);
  }
  L.push(`| overall | **${pct(a.flipAll)}** |`);
  L.push("");
  L.push(
    a.flippedItems.length
      ? `${a.flippedItems.length} item(s) changed decision at least once: ` +
        `${a.flippedItems.join(", ")}.`
      : "No item ever changed its decision.",
  );
  L.push("");

  if (a.warnings.length) {
    L.push("### Verdict");
    L.push("");
    for (const w of a.warnings) L.push(`- **${w}**`);
    L.push("");
  } else {
    L.push("### Verdict");
    L.push("");
    L.push("- No blocking issues found for this design.");
    L.push("");
  }

  L.push("### 1. Repeatability");
  L.push("");
  L.push(
    `Identical requests: **${a.deterministicItems}** item(s) returned a ` +
      `byte-identical distribution every time, **${a.jitterItems}** did not.`,
  );
  if (a.repeatSd !== null) {
    L.push(`Pooled within-cell sd: **${f4(a.repeatSd)}**.`);
  }
  L.push(
    `Modal answer moved for **${a.modalMoves}** of ` +
      `${Object.keys(a.itemMeans).length} items.`,
  );
  L.push("");

  if (a.nullSd !== null || a.paraSd !== null) {
    L.push("### 2. Framing sensitivity");
    L.push("");
    L.push("| source | sd |");
    L.push("|---|---|");
    L.push(`| semantically-null change | ${f4(a.nullSd)} |`);
    L.push(`| paraphrase | ${f4(a.paraSd)} |`);
    L.push(`| repeat noise | ${f4(a.repeatSd)} |`);
    L.push("");
    if (a.driftRatio) {
      L.push(
        `Paraphrase moves the answer **${a.driftRatio.toFixed(1)}×** more ` +
          `than a null change.`,
      );
      L.push("");
    }
  }

  L.push("### 3. Resolution — can this question rank your items?");
  L.push("");
  L.push(
    `**${a.distinctMeans}** distinct values across ` +
      `${Object.keys(a.itemMeans).length} items.`,
  );
  if (a.resolution !== null) {
    L.push(
      `Two items are separable only if they differ by more than ` +
        `**${a.resolution.toFixed(3)}**.`,
    );
    L.push(
      `Indistinguishable pairs: **${a.indistinguishablePairs}/${a.totalPairs}**` +
        ` (${(100 * a.indistinguishablePairs / Math.max(a.totalPairs, 1))
          .toFixed(1)}%).`,
    );
  }
  if (a.exactTies.length) {
    L.push("");
    L.push("Exact ties (identical value — unrankable):");
    L.push("");
    for (const t of a.exactTies) {
      L.push(`- \`${t.value.toFixed(4)}\` — ${t.items.join(", ")}`);
    }
  }
  L.push("");

  if (a.thresholdFlappers.length) {
    L.push("### 4. Threshold stability");
    L.push("");
    L.push("| item | confidence range |");
    L.push("|---|---|");
    for (const t of a.thresholdFlappers) {
      L.push(`| ${t.item} | ${t.lo.toFixed(3)} – ${t.hi.toFixed(3)} |`);
    }
    L.push("");
    L.push(
      "These straddle a common cut-off **on identical input**. A fixed " +
        "threshold flips between runs.",
    );
    L.push("");
  }

  if (a.answerability) {
    L.push("### 5. Answerability");
    L.push("");
    L.push(
      `Asked for ${a.answerability.n} items; range ` +
        `${a.answerability.min.toFixed(3)} – ${a.answerability.max.toFixed(3)}.`,
    );
    if (a.answerability.lowAnswerHighConf.length) {
      L.push("");
      L.push("**Confident but nothing to judge:**");
      L.push("");
      L.push("| item | confidence | answerable |");
      L.push("|---|---|---|");
      for (const x of a.answerability.lowAnswerHighConf) {
        L.push(
          `| ${x.item} | ${x.conf.toFixed(3)} | ${x.ans.toFixed(3)} |`,
        );
      }
    }
    L.push("");
  }

  if (a.fit) {
    L.push("### 6. Bayesian variance decomposition");
    L.push("");
    L.push(
      `Nested measurement model (repeat → framing → item), 4 chains × ` +
        `${a.fit.iters} iterations. **${a.saturatedExcluded} saturated item(s) ` +
        `excluded** — pinned at a scale end, they carry no variance ` +
        `information and inflate the between-item term. These components ` +
        `therefore describe the unsaturated region only.`,
    );
    L.push("");
    L.push(
      `Convergence gate: max R̂ **${a.fit.maxRhat.toFixed(4)}** (< 1.01), ` +
        `min ESS **${a.fit.minEss.toFixed(0)}** (> 400) — ` +
        `**${a.fit.converged ? "PASS" : "FAIL"}**`,
    );
    L.push("");
    if (a.fit.converged) {
      L.push("| component | mean | 95% CI | R̂ | ESS |");
      L.push("|---|---|---|---|---|");
      for (const r of a.fit.rows) {
        L.push(
          `| ${LABEL[r.param] ?? r.param} | ${r.mean.toFixed(5)} | ` +
            `[${r.lo.toFixed(5)}, ${r.hi.toFixed(5)}] | ` +
            `${r.rhat.toFixed(4)} | ${r.ess.toFixed(0)} |`,
        );
      }
      const drift = a.fit.rows.find((r) => r.param === "drift")!;
      L.push("");
      L.push(
        drift.lo > 0
          ? `Construct drift is **established**: the 95% interval ` +
            `[${drift.lo.toFixed(5)}, ${drift.hi.toFixed(5)}] excludes zero. ` +
            `Rewording this question changes what it measures.`
          : `Construct drift is **not established**: the 95% interval ` +
            `includes zero.`,
      );
    } else {
      L.push(
        "_Posterior numbers withheld — a non-converged fit must not be " +
          "interpreted._",
      );
    }
    L.push("");
  }

  L.push("### What this cannot tell you");
  L.push("");
  L.push(
    "- **Nothing about accuracy.** This measures whether the question is " +
      "*consistent*, never whether it is *right*. A perfectly repeatable " +
      "question can be repeatably wrong. Checking that needs labelled data.",
  );
  L.push(
    "- Results apply to this question, this model version and these items " +
      "only.",
  );
  L.push("");
  return L.join("\n");
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

const dec = new TextDecoder();

export const report = {
  name: "@vcjdeboer/jev-reliability-report",
  description:
    "Pre-flight verdict for a Jev question. Reads every replicate-* grid " +
    "and reports repeatability, framing sensitivity, resolution, threshold " +
    "stability, answerability and a nested Bayesian variance decomposition — " +
    "withholding posterior numbers when the fit does not converge.",
  scope: "model" as const,
  labels: ["jev", "typesafe", "reliability", "validation"],
  execute: async (context: {
    modelType: { type: string };
    modelId: string;
    dataRepository: {
      getContent(
        type: { type: string },
        modelId: string,
        dataName: string,
        version?: number,
      ): Promise<Uint8Array | null>;
      findAllForModel(
        type: { type: string },
        modelId: string,
      ): Promise<DataEntry[]>;
    };
    definition: { name: string };
    logger?: { info(msg: string, meta?: unknown): void };
  }): Promise<{ markdown: string; json: Record<string, unknown> }> => {
    const { modelType, modelId, dataRepository, logger } = context;

    const all = await dataRepository.findAllForModel(modelType, modelId);
    const entries = all.filter(
      (d) => d.tags?.specName === "replicate" || d.name.startsWith("replicate-"),
    );
    logger?.info(`Found ${entries.length} replicate grids`, {
      names: entries.map((e) => e.name),
    });

    // Iterations come from the grid (a documented method argument), with an
    // env override for one-off experiments.
    const ITER_OVERRIDE = Deno.env.get("JEV_REPORT_ITERS");

    const analyses: Analysis[] = [];
    for (const entry of entries) {
      const raw = await dataRepository.getContent(
        modelType,
        modelId,
        entry.name,
        entry.version,
      );
      if (!raw) continue;
      const grid = JSON.parse(dec.decode(raw)) as Grid;
      // Reports fire after a method run, so a tiny run is used to trigger
      // them. Those grids carry no information — skip rather than render an
      // empty section for them.
      if (!grid.observations?.length) continue;
      if (entry.name.endsWith("-report-trigger") ||
        grid.observations.length < 4) {
        logger?.info(`Skipping trigger grid ${grid.run}`);
        continue;
      }
      logger?.info(`Analysing ${grid.run}`, {
        calls: grid.observations.length,
      });
      const iters = ITER_OVERRIDE
        ? Number(ITER_OVERRIDE)
        : grid.fitIterations ?? 40000;
      analyses.push(analyse(grid, iters));
    }

    analyses.sort((a, b) => a.collectedAt.localeCompare(b.collectedAt));

    const md: string[] = [
      "# Jev question pre-flight",
      "",
      "Is this question safe to build on? Each section answers one thing you " +
      "need settled before shipping a Jev-backed feature.",
      "",
    ];
    if (!analyses.length) {
      md.push("_No replicate grids found. Run the `replicate` method first._");
    } else {
      for (const a of analyses) md.push(renderOne(a));
    }

    return {
      markdown: md.join("\n"),
      json: {
        generatedAt: new Date().toISOString(),
        grids: analyses.length,
        runs: analyses.map((a) => ({
          run: a.run,
          model: a.model,
          questionType: a.questionType,
          flipRateStable: a.flipStable,
          flipRateParaphrase: a.flipPara,
          flipRateOverall: a.flipAll,
          flippedItems: a.flippedItems,
          noopFramings: a.noopFramings,
          calls: a.design.totalCalls,
          ok: a.ok,
          failed: a.failed,
          repeatSd: a.repeatSd,
          nullSd: a.nullSd,
          paraSd: a.paraSd,
          driftRatio: a.driftRatio,
          distinctMeans: a.distinctMeans,
          exactlyTiedItems: a.exactTies.reduce((s, t) => s + t.items.length, 0),
          resolution: a.resolution,
          indistinguishablePairs: a.indistinguishablePairs,
          totalPairs: a.totalPairs,
          thresholdFlappers: a.thresholdFlappers.length,
          answerabilityAsked: a.answerability !== null,
          confidentButUnanswerable:
            a.answerability?.lowAnswerHighConf.length ?? 0,
          fit: a.fit
            ? {
              converged: a.fit.converged,
              maxRhat: a.fit.maxRhat,
              minEss: a.fit.minEss,
              components: a.fit.converged
                ? Object.fromEntries(
                  a.fit.rows.map((
                    r,
                  ) => [r.param, { mean: r.mean, lo: r.lo, hi: r.hi }]),
                )
                : null,
            }
            : null,
          warnings: a.warnings,
        })),
      },
    };
  },
};
