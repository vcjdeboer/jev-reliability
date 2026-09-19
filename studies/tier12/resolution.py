"""How much of this corpus can Jev actually rank?

Tier 1/2 gave a paraphrase-level sd. Two items are only separable if their
means differ by more than the noise in estimating that difference. This script
turns the variance components into a practical resolution figure and counts how
many item pairs fall inside it.
"""

import json
import statistics as st
import sys

RAW = sys.argv[1] if len(sys.argv) > 1 else "studies/tier12/raw.json"


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


obs = [o for o in find_observations(json.load(open(RAW)))
       if o["score"] is not None]

by_item = {}
for o in obs:
    by_item.setdefault(o["item"], []).append(o["score"])
means = {i: st.mean(v) for i, v in by_item.items()}

exact = {}
for i, m in means.items():
    exact.setdefault(round(m, 6), []).append(i)

print("distinct item means:", len(exact), "across", len(means), "items")
print("\nexact ties (identical mean to 6dp):")
for m, ids in sorted(exact.items()):
    if len(ids) > 1:
        print(f"  {m:.4f}  n={len(ids):2d}  {', '.join(sorted(ids))}")

# Paraphrase sd is the dominant non-item component; a difference of two
# independent item estimates carries sqrt(2) times it.
# Corrected collection (prereg D9). The superseded value was 0.0371.
SD_PARA = 0.0347
resolution = 1.96 * (2 ** 0.5) * SD_PARA
print(f"\nparaphrase sd = {SD_PARA:.4f}")
print(f"95% resolution on a difference = 1.96*sqrt(2)*sd = {resolution:.4f}")

ids = sorted(means)
pairs = [(a, b) for k, a in enumerate(ids) for b in ids[k + 1:]]
indist = [(a, b) for a, b in pairs if abs(means[a] - means[b]) < resolution]
print(f"\nitem pairs: {len(pairs)}")
print(f"indistinguishable at 95%: {len(indist)} "
      f"({100*len(indist)/len(pairs):.1f}%)")

mid = [i for i in ids if i.startswith("m")]
mpairs = [(a, b) for k, a in enumerate(mid) for b in mid[k + 1:]]
mind = [(a, b) for a, b in mpairs if abs(means[a] - means[b]) < resolution]
print(f"mid-stratum pairs: {len(mpairs)}, indistinguishable: {len(mind)} "
      f"({100*len(mind)/len(mpairs):.1f}%)")

span = max(means[i] for i in mid) - min(means[i] for i in mid)
print(f"\nmid-stratum span: {span:.3f} -> "
      f"~{span/resolution:.1f} resolvable bands")
