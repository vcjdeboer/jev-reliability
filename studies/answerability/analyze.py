"""Study B — does a separate answerability Noul detect what confidence cannot?

Pre-registered (prereg 8b):
  H1  mean answerable lower for unjudgeable than judgeable
  H2  answerable separates the classes BETTER than confidence does

Falsification: if AUROC(confidence) is comparable to AUROC(answerable), the
Score's own confidence already carries the signal and the separate question
buys nothing.
"""

import json
import statistics as st
import sys

RAW = sys.argv[1] if len(sys.argv) > 1 else "studies/answerability/raw.json"


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


def auroc(pos, neg):
    """Mann-Whitney U / rank-based AUROC, ties counted as half."""
    wins = sum(
        1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg
    )
    return wins / (len(pos) * len(neg))


obs = find_observations(json.load(open(RAW)))
print(f"observations: {len(obs)}")

# --- did the Noul parse at all? -------------------------------------------
sample = next((o for o in obs if o.get("answerableProbs") is not None
               or o.get("answerable") is not None), None)
print("\nraw Noul shape (first observation):")
print("  answerable        =", sample and sample["answerable"])
print("  answerableConf    =", sample and sample["answerableConfidence"])
print("  answerableProbs   =", sample and sample["answerableProbs"])
n_parsed = sum(1 for o in obs if o.get("answerable") is not None)
print(f"  parsed: {n_parsed}/{len(obs)}")
if n_parsed == 0:
    print("\nNoul did not parse — cannot evaluate H1/H2.")
    sys.exit(1)

# --- per item --------------------------------------------------------------
items = {}
for o in obs:
    items.setdefault(o["item"], []).append(o)

rows = []
for item in sorted(items):
    v = items[item]
    rows.append({
        "item": item,
        "cls": "judgeable" if item.startswith("j") else "unjudgeable",
        "answerable": st.mean([o["answerable"] for o in v
                               if o["answerable"] is not None]),
        "confidence": st.mean([o["confidence"] for o in v
                               if o["confidence"] is not None]),
        "score": st.mean([o["score"] for o in v if o["score"] is not None]),
    })

print(f"\n{'item':6} {'class':12} {'answerable':>11} {'confidence':>11} "
      f"{'score':>7}")
for r in rows:
    print(f"{r['item']:6} {r['cls']:12} {r['answerable']:11.4f} "
          f"{r['confidence']:11.4f} {r['score']:7.3f}")

J = [r for r in rows if r["cls"] == "judgeable"]
U = [r for r in rows if r["cls"] == "unjudgeable"]

print(f"\n{'metric':12} {'judgeable':>12} {'unjudgeable':>12} {'gap':>8} "
      f"{'AUROC':>7}")
for m in ("answerable", "confidence"):
    mj = st.mean([r[m] for r in J])
    mu = st.mean([r[m] for r in U])
    a = auroc([r[m] for r in J], [r[m] for r in U])
    print(f"{m:12} {mj:12.4f} {mu:12.4f} {mj-mu:8.4f} {a:7.3f}")

# --- H2 verdict ------------------------------------------------------------
a_ans = auroc([r["answerable"] for r in J], [r["answerable"] for r in U])
a_con = auroc([r["confidence"] for r in J], [r["confidence"] for r in U])
print(f"\nH2: AUROC(answerable)={a_ans:.3f} vs AUROC(confidence)={a_con:.3f}"
      f"  -> delta {a_ans - a_con:+.3f}")

# --- the dissociation ------------------------------------------------------
print("\nDissociation — confident but unanswerable (the case a stability "
      "gate cannot catch):")
hits = [r for r in rows
        if r["confidence"] >= 0.8 and r["answerable"] <= 0.5]
for r in hits:
    print(f"  {r['item']:6} {r['cls']:12} conf={r['confidence']:.3f} "
          f"answerable={r['answerable']:.3f} score={r['score']:.3f}")
print(f"  count: {len(hits)}/{len(rows)}")
