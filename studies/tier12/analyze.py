"""Descriptive nested variance decomposition for the Tier 1/2 replicate grid.

Moment-based ANOVA pass, run before any Bayesian fit. It answers the two
pre-registered questions cheaply and gives the Stan model sensible starting
expectations:

  RQ2  sigma^2_null                  -- instrument noise
  RQ3  sigma^2_para - sigma^2_null   -- construct drift
  RQ4  G = var_item / (var_item + var_null + var_resid)

Cell means of n repeats carry var_repeat/n of sampling noise, so the raw
between-framing variance is de-biased by subtracting it. Without that, the
framing components are inflated by the repeat noise Tier 0 already measured.
"""

import json
import statistics as st
import sys

RAW = sys.argv[1] if len(sys.argv) > 1 else "studies/tier12/raw.json"


def find_observations(node):
    """swamp nests the resource payload; locate the observations array."""
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


def stratum(item_id):
    return {"c": "casual", "m": "mid", "f": "formal"}[item_id[0]]


raw = json.load(open(RAW))
obs = [o for o in find_observations(raw) if o["score"] is not None]
print(f"observations: {len(obs)}")

# --- cells -----------------------------------------------------------------
cells = {}
for o in obs:
    cells.setdefault((o["item"], o["framing"], o["framingType"]), []).append(
        o["score"]
    )
n_per_cell = st.mode([len(v) for v in cells.values()])

# --- 1. repeat / residual variance (pooled within cell) --------------------
within = [st.pvariance(v) for v in cells.values() if len(v) > 1]
var_rep = sum(within) / len(within)

# --- 2. framing variance within item, by type ------------------------------
items = {}
for (item, _framing, ftype), scores in cells.items():
    items.setdefault(item, {}).setdefault(ftype, []).append(
        sum(scores) / len(scores)
    )

var_framing = {}
for ftype in ("null", "para"):
    per_item = [
        st.pvariance(means)
        for means in (items[i][ftype] for i in items)
        if len(means) > 1
    ]
    rawv = sum(per_item) / len(per_item)
    var_framing[ftype] = {
        "raw": rawv,
        "debiased": max(rawv - var_rep / n_per_cell, 0.0),
    }

# --- 3. item variance ------------------------------------------------------
item_means = {
    i: st.mean([m for ft in items[i] for m in items[i][ft]]) for i in items
}
var_item = st.pvariance(list(item_means.values()))

# --- report ----------------------------------------------------------------
print(f"\nrepeats per cell: {n_per_cell}\n")
print(f"{'component':14} {'variance':>12} {'sd':>10}")
print(f"{'item':14} {var_item:12.6f} {var_item**0.5:10.4f}")
for ftype in ("null", "para"):
    d = var_framing[ftype]["debiased"]
    print(f"{'framing_'+ftype:14} {d:12.6f} {d**0.5:10.4f}   "
          f"(raw {var_framing[ftype]['raw']:.6f})")
print(f"{'repeat':14} {var_rep:12.6f} {var_rep**0.5:10.4f}")

v_null = var_framing["null"]["debiased"]
v_para = var_framing["para"]["debiased"]
drift = v_para - v_null
print(f"\nRQ3 construct drift  sigma^2_para - sigma^2_null = {drift:.6f}"
      f"  (sd-scale {abs(drift)**0.5:.4f})")

g = var_item / (var_item + v_null + var_rep)
print(f"RQ4 G (hard-region, per D5) = {g:.4f}")

print(f"\nitem mean range: {min(item_means.values()):.3f} .. "
      f"{max(item_means.values()):.3f}")

# --- per stratum -----------------------------------------------------------
print(f"\n{'stratum':10} {'n':>3} {'var_item':>10} {'var_null':>10} "
      f"{'var_para':>10} {'var_rep':>10}")
for s in ("casual", "mid", "formal"):
    ids = [i for i in items if stratum(i) == s]
    sv_item = st.pvariance([item_means[i] for i in ids])
    sv_rep_parts = [
        st.pvariance(v) for (it, _f, _t), v in cells.items()
        if stratum(it) == s and len(v) > 1
    ]
    sv_rep = sum(sv_rep_parts) / len(sv_rep_parts)
    row = [sv_item]
    for ftype in ("null", "para"):
        parts = [st.pvariance(items[i][ftype]) for i in ids
                 if len(items[i][ftype]) > 1]
        row.append(max(sum(parts) / len(parts) - sv_rep / n_per_cell, 0.0))
    print(f"{s:10} {len(ids):3d} {row[0]:10.5f} {row[1]:10.5f} "
          f"{row[2]:10.5f} {sv_rep:10.5f}")

# --- did the modal answer move? -------------------------------------------
argmax = {}
for o in obs:
    p = o["probabilities"] or {}
    if p:
        argmax.setdefault(o["item"], {}).setdefault(
            o["framingType"], set()
        ).add(max(p, key=p.get))

flips = {
    ft: sum(1 for i in argmax if len(argmax[i].get(ft, set())) > 1)
    for ft in ("null", "para")
}
print(f"\nitems whose modal level moved: null={flips['null']}/30  "
      f"para={flips['para']}/30")
overall = sum(
    1 for i in argmax
    if len(set().union(*argmax[i].values())) > 1
)
print(f"items whose modal level moved across ANY framing: {overall}/30")
