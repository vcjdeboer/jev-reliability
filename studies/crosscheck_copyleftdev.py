"""Cross-check our variance components against copyleftdev/jev-labs.

That write-up reports, over 1,490 captured jev-1.13.0 calls, three "noise
floors" on the 0-1 probability scale:

    identity 0.042 · question-reorder 0.059 · paraphrase cohort 0.073

Our components are on the 0-3 `score` scale, so they are not directly
comparable. This rescales our data onto the probability scale and reports
both sd and max spread, because their "floor" is a gate margin (a bound),
not a standard deviation.
"""

import json
import statistics as st

SCALES = {
    "tier0": "studies/tier0/raw.json",
    # Corrected collection: the original had key_reorder byte-identical to
# identity (prereg D9), which biased the null cohort toward repeat noise.
    "tier12": "studies/tier12/raw-fixed.json",
}


def find_observations(node):
    if isinstance(node, dict):
        if "observations" in node:
            return node["observations"]
        for value in node.values():
            found = find_observations(value)
            if found:
                return found
    if isinstance(node, list):
        for value in node:
            found = find_observations(value)
            if found:
                return found
    return None


def load(path):
    return [o for o in find_observations(json.load(open(path)))
            if o.get("probabilities")]


def spread_stats(groups):
    """groups: list of lists of probability dicts judged identical-by-design."""
    sds, ranges = [], []
    for g in groups:
        if len(g) < 2:
            continue
        levels = sorted(g[0].keys())
        for lv in levels:
            vals = [d[lv] for d in g]
            sds.append(st.pstdev(vals))
            ranges.append(max(vals) - min(vals))
    return sds, ranges


def report(label, groups):
    sds, ranges = spread_stats(groups)
    if not sds:
        print(f"{label:28} (no replicated cells)")
        return
    print(f"{label:28} n_cells={len(sds):4d}  "
          f"mean_sd={st.mean(sds):.4f}  max_sd={max(sds):.4f}  "
          f"mean_range={st.mean(ranges):.4f}  max_range={max(ranges):.4f}")


print("All figures on the 0-1 PROBABILITY scale (per level), "
      "to match the article.\n")

# --- Tier 0: pure identity, 20 repeats per item ---------------------------
t0 = load(SCALES["tier0"])
g = {}
for o in t0:
    g.setdefault(o["item"], []).append(o["probabilities"])
report("tier0 identity (20 reps)", list(g.values()))

# Tier 0 excluding the two fully saturated items
g_unsat = {k: v for k, v in g.items() if k not in
           ("i1_casual", "i5_formal")}
report("tier0 identity, unsaturated", list(g_unsat.values()))

# --- Tier 1/2 -------------------------------------------------------------
t12 = load(SCALES["tier12"])

# identity only, 2 repeats
ident = {}
for o in t12:
    if o["framing"] == "identity":
        ident.setdefault(o["item"], []).append(o["probabilities"])
report("tier12 identity (2 reps)", list(ident.values()))

# across the 4 null framings (pooling their repeats)
null = {}
for o in t12:
    if o["framingType"] == "null":
        null.setdefault(o["item"], []).append(o["probabilities"])
report("tier12 null cohort (8)", list(null.values()))

# across the 4 paraphrases
para = {}
for o in t12:
    if o["framingType"] == "para":
        para.setdefault(o["item"], []).append(o["probabilities"])
report("tier12 paraphrase cohort (8)", list(para.values()))

# unsaturated subset only (mid stratum) — where the article's items likely sit
para_mid = {k: v for k, v in para.items() if k.startswith("m")}
null_mid = {k: v for k, v in null.items() if k.startswith("m")}
print()
report("tier12 null cohort, mid only", list(null_mid.values()))
report("tier12 para cohort, mid only", list(para_mid.values()))

print("\narticle reference: identity 0.042 · reorder 0.059 · paraphrase 0.073")
