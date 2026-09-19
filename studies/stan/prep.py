"""Emit the Tier 1/2 mid-stratum cells as a flat CSV for the Stan fit.

Mid stratum only, per prereg D6: the saturated items are deterministic at the
scale boundary and carry no information about variance components.
"""

import csv
import json
import sys

RAW = sys.argv[1] if len(sys.argv) > 1 else "studies/tier12/raw.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "studies/stan/data.csv"


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


obs = [
    o for o in find_observations(json.load(open(RAW)))
    if o["score"] is not None and o["item"].startswith("m")
]

items = sorted({o["item"] for o in obs})
framings = sorted({o["framing"] for o in obs})
ftype = {o["framing"]: o["framingType"] for o in obs}

with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["score", "item", "framing", "is_para"])
    for o in obs:
        w.writerow([
            o["score"],
            items.index(o["item"]) + 1,
            framings.index(o["framing"]) + 1,
            1 if o["framingType"] == "para" else 0,
        ])

print(f"rows={len(obs)} items={len(items)} framings={len(framings)}")
print("framing index -> type:")
for i, f in enumerate(framings, 1):
    print(f"  {i}: {f} ({ftype[f]})")
