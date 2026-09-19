"""Gibbs / slice sampler for the nested measurement model.

Stan could not be used: rstan's C++ toolchain on this machine fails to compile
any model (Apple libc++ headers expect a newer clang than is installed;
RcppEigen is already the fixed version, so it is a system fault, not a package
one). cmdstanr is not installed and there is no Python scientific stack.

Rather than patch someone's system toolchain, the model is fitted with a
hand-written sampler using only the standard library. That is only acceptable
if the sampler is shown to work, so `validate` fits simulated data with known
parameters and checks recovery before any real data is touched.

Model (prereg section 5 as amended by D6):

    y[n] = mu + a[item] + f[fram] + g[item,fram] + eps
    a ~ N(0, s_item)      f ~ N(0, s_fram)
    g[i,j] ~ N(0, s_null) if framing j is null, N(0, s_para) if paraphrase
    eps ~ N(0, sigma_y)
    mu ~ N(1.5, 1);  every s ~ half-N(0, prior_scale)

All location parameters have conjugate Gaussian full conditionals. The five
scale parameters are slice-sampled, which needs no tuning.
"""

import csv
import math
import random
import sys


# --------------------------------------------------------------------------
# sampling primitives
# --------------------------------------------------------------------------

def slice_sample(x0, logp, w=0.05, lower=1e-9, upper=10.0, max_step=50):
    """Univariate slice sampler with stepping out (Neal 2003)."""
    y = logp(x0) + math.log(random.random())
    lo = max(lower, x0 - w * random.random())
    hi = min(upper, lo + w)
    steps = 0
    while lo > lower and logp(lo) > y and steps < max_step:
        lo = max(lower, lo - w)
        steps += 1
    steps = 0
    while hi < upper and logp(hi) > y and steps < max_step:
        hi = min(upper, hi + w)
        steps += 1
    for _ in range(100):
        x = lo + random.random() * (hi - lo)
        if logp(x) > y:
            return x
        if x < x0:
            lo = x
        else:
            hi = x
    return x0


def scale_logpost(vals, s, prior_scale):
    """half-N(0, prior_scale) prior x Normal(0, s) likelihood over `vals`."""
    if s <= 0:
        return -math.inf
    n = len(vals)
    ss = sum(v * v for v in vals)
    return (-0.5 * (s * s) / (prior_scale * prior_scale)
            - n * math.log(s) - 0.5 * ss / (s * s))


# --------------------------------------------------------------------------
# the sampler
# --------------------------------------------------------------------------

def run_chain(y, item, fram, is_para, n_items, n_fram, prior_scale,
              iters, burn, seed):
    random.seed(seed)
    N = len(y)

    cell_obs = {}
    for n in range(N):
        cell_obs.setdefault((item[n], fram[n]), []).append(n)
    item_obs = {}
    fram_obs = {}
    for n in range(N):
        item_obs.setdefault(item[n], []).append(n)
        fram_obs.setdefault(fram[n], []).append(n)

    mu = sum(y) / N
    a = [0.0] * n_items
    f = [0.0] * n_fram
    g = {k: 0.0 for k in cell_obs}
    s_item = s_fram = s_null = s_para = sigma_y = 0.1

    draws = []
    for it in range(iters):
        # ---- mu -----------------------------------------------------------
        prec = 1.0 / 1.0 + N / (sigma_y ** 2)
        acc = 1.5 / 1.0
        for n in range(N):
            acc += (y[n] - a[item[n]] - f[fram[n]]
                    - g[(item[n], fram[n])]) / (sigma_y ** 2)
        mu = acc / prec + random.gauss(0, 1) / math.sqrt(prec)

        # ---- a (items) ------------------------------------------------------
        for i in range(n_items):
            idx = item_obs[i]
            prec = 1.0 / (s_item ** 2) + len(idx) / (sigma_y ** 2)
            acc = sum(y[n] - mu - f[fram[n]] - g[(item[n], fram[n])]
                      for n in idx) / (sigma_y ** 2)
            a[i] = acc / prec + random.gauss(0, 1) / math.sqrt(prec)

        # ---- f (framing main effect) ----------------------------------------
        for j in range(n_fram):
            idx = fram_obs[j]
            prec = 1.0 / (s_fram ** 2) + len(idx) / (sigma_y ** 2)
            acc = sum(y[n] - mu - a[item[n]] - g[(item[n], fram[n])]
                      for n in idx) / (sigma_y ** 2)
            f[j] = acc / prec + random.gauss(0, 1) / math.sqrt(prec)

        # ---- g (item x framing) ---------------------------------------------
        for (i, j), idx in cell_obs.items():
            s_g = s_para if is_para[j] else s_null
            prec = 1.0 / (s_g ** 2) + len(idx) / (sigma_y ** 2)
            acc = sum(y[n] - mu - a[i] - f[j] for n in idx) / (sigma_y ** 2)
            g[(i, j)] = acc / prec + random.gauss(0, 1) / math.sqrt(prec)

        # ---- ridge-breaking joint shifts ------------------------------------
        # mu and the group effects are identified only through their priors:
        # (mu + d, a - d) has exactly the same likelihood. Gibbs therefore
        # crawls along that ridge and mu fails to mix. These two moves jump
        # along it directly. The proposal is symmetric and the likelihood is
        # invariant, so the acceptance ratio involves only the priors.
        for vec, s_vec in ((a, s_item), (f, s_fram)):
            d = random.gauss(0, 0.05)
            logr = (
                -0.5 * ((mu + d - 1.5) ** 2 - (mu - 1.5) ** 2)
                + sum(-0.5 * ((v - d) ** 2 - v * v) / (s_vec ** 2)
                      for v in vec)
            )
            if math.log(random.random()) < logr:
                mu += d
                for k in range(len(vec)):
                    vec[k] -= d

        # ---- scales ---------------------------------------------------------
        s_item = slice_sample(
            s_item, lambda s: scale_logpost(a, s, prior_scale))
        s_fram = slice_sample(
            s_fram, lambda s: scale_logpost(f, s, prior_scale))
        gn = [v for (i, j), v in g.items() if not is_para[j]]
        gp = [v for (i, j), v in g.items() if is_para[j]]
        s_null = slice_sample(
            s_null, lambda s: scale_logpost(gn, s, prior_scale))
        s_para = slice_sample(
            s_para, lambda s: scale_logpost(gp, s, prior_scale))
        resid = [y[n] - mu - a[item[n]] - f[fram[n]] - g[(item[n], fram[n])]
                 for n in range(N)]
        sigma_y = slice_sample(
            sigma_y, lambda s: scale_logpost(resid, s, prior_scale))

        if it >= burn:
            draws.append({
                "mu": mu, "s_item": s_item, "s_fram": s_fram,
                "s_null": s_null, "s_para": s_para, "sigma_y": sigma_y,
                "drift": s_para ** 2 - s_null ** 2,
                "G": s_item ** 2 / (s_item ** 2 + s_null ** 2 + sigma_y ** 2),
            })
    return draws


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------

def split_rhat(chains):
    """Split-Rhat (Vehtari et al. 2021) on a list of equal-length chains."""
    halves = []
    for c in chains:
        h = len(c) // 2
        halves.extend([c[:h], c[h:2 * h]])
    m, n = len(halves), len(halves[0])
    means = [sum(h) / n for h in halves]
    grand = sum(means) / m
    B = n * sum((mk - grand) ** 2 for mk in means) / (m - 1)
    W = sum(sum((x - means[k]) ** 2 for x in halves[k]) / (n - 1)
            for k in range(m)) / m
    if W <= 0:
        return float("nan")
    var_hat = (n - 1) / n * W + B / n
    return math.sqrt(var_hat / W)


def ess(chains):
    """Effective sample size via Geyer's initial positive sequence."""
    m, n = len(chains), len(chains[0])
    means = [sum(c) / n for c in chains]
    grand = sum(means) / m
    W = sum(sum((x - means[k]) ** 2 for x in chains[k]) / (n - 1)
            for k in range(m)) / m
    if W <= 0:
        return float("nan")

    def acov(lag):
        tot = 0.0
        for k, c in enumerate(chains):
            tot += sum((c[t] - means[k]) * (c[t + lag] - means[k])
                       for t in range(n - lag)) / n
        return tot / m

    var0 = acov(0)
    if var0 <= 0:
        return float("nan")
    rho, t = [], 1
    while t < n - 2:
        p = acov(t) / var0 + acov(t + 1) / var0
        if p <= 0:
            break
        rho.append(p)
        t += 2
    tau = 1 + 2 * sum(rho)
    B = n * sum((mk - grand) ** 2 for mk in means) / (m - 1) if m > 1 else 0
    var_hat = (n - 1) / n * W + B / n
    return m * n * (var_hat / var0) / max(tau, 1e-9)


def summarise(all_chains, params):
    print(f"{'param':10} {'mean':>10} {'2.5%':>10} {'97.5%':>10} "
          f"{'Rhat':>8} {'ESS':>8}")
    worst_rhat, worst_ess = 0.0, math.inf
    for p in params:
        chains = [[d[p] for d in c] for c in all_chains]
        flat = sorted(x for c in chains for x in c)
        mean = sum(flat) / len(flat)
        lo = flat[int(0.025 * len(flat))]
        hi = flat[int(0.975 * len(flat))]
        r = split_rhat(chains)
        e = ess(chains)
        worst_rhat = max(worst_rhat, 0 if math.isnan(r) else r)
        worst_ess = min(worst_ess, math.inf if math.isnan(e) else e)
        print(f"{p:10} {mean:10.5f} {lo:10.5f} {hi:10.5f} {r:8.4f} {e:8.0f}")
    return worst_rhat, worst_ess


PARAMS = ["mu", "s_item", "s_fram", "s_null", "s_para", "sigma_y",
          "drift", "G"]


def fit(y, item, fram, is_para, n_items, n_fram, prior_scale=1.0,
        iters=4000, burn=2000, chains=4):
    return [run_chain(y, item, fram, is_para, n_items, n_fram, prior_scale,
                      iters, burn, seed=1000 + k) for k in range(chains)]


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------

def load(path):
    y, item, fram, ip = [], [], [], {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            y.append(float(row["score"]))
            item.append(int(row["item"]) - 1)
            fram.append(int(row["framing"]) - 1)
            ip[int(row["framing"]) - 1] = int(row["is_para"]) == 1
    n_fram = max(fram) + 1
    return y, item, fram, [ip[j] for j in range(n_fram)], max(item) + 1, n_fram


def validate():
    """Fit simulated data with known parameters and check recovery."""
    print("=== sampler validation: recovery on simulated data ===")
    random.seed(7)
    truth = dict(mu=1.3, s_item=0.45, s_fram=0.02, s_null=0.015,
                 s_para=0.055, sigma_y=0.016)
    n_items, n_fram, reps = 14, 8, 2
    is_para = [False, False, False, True, True, True, True, False]
    a = [random.gauss(0, truth["s_item"]) for _ in range(n_items)]
    f = [random.gauss(0, truth["s_fram"]) for _ in range(n_fram)]
    y, item, fram = [], [], []
    for i in range(n_items):
        for j in range(n_fram):
            gij = random.gauss(
                0, truth["s_para"] if is_para[j] else truth["s_null"])
            for _ in range(reps):
                y.append(truth["mu"] + a[i] + f[j] + gij
                         + random.gauss(0, truth["sigma_y"]))
                item.append(i)
                fram.append(j)
    ch = fit(y, item, fram, is_para, n_items, n_fram)
    r, e = summarise(ch, PARAMS)
    print("\ntruth:", {k: round(v, 4) for k, v in truth.items()})
    print(f"worst Rhat {r:.4f} | min ESS {e:.0f}")
    ok = True
    for p in ["s_item", "s_null", "s_para", "sigma_y"]:
        flat = sorted(x for c in ch for x in (d[p] for d in c))
        lo = flat[int(0.025 * len(flat))]
        hi = flat[int(0.975 * len(flat))]
        covered = lo <= truth[p] <= hi
        ok &= covered
        print(f"  {p:9} truth={truth[p]:.4f} in [{lo:.4f}, {hi:.4f}] "
              f"-> {'COVERED' if covered else 'MISSED'}")
    print("recovery:", "PASS" if ok else "FAIL")
    return ok


def main():
    if not validate():
        print("\nSampler failed recovery — not fitting real data.")
        sys.exit(1)

    path = sys.argv[1] if len(sys.argv) > 1 else "studies/stan/data.csv"
    y, item, fram, is_para, n_items, n_fram = load(path)
    print(f"\n\n=== real data: N={len(y)} items={n_items} "
          f"framings={n_fram} ===")

    # s_null and sigma_y are only weakly separable with 2 repeats per cell, so
    # that pair mixes slowly and needs a long run to clear the ESS gate.
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 60000
    burn = iters // 4
    print(f"(main fit: {iters} iterations, {burn} burn-in, 4 chains)")
    ch = fit(y, item, fram, is_para, n_items, n_fram, prior_scale=1.0,
             iters=iters, burn=burn)
    r, e = summarise(ch, PARAMS)
    print(f"\nconvergence gate (prereg section 6): "
          f"max Rhat {r:.4f} {'PASS' if r < 1.01 else 'FAIL'} | "
          f"min ESS {e:.0f} {'PASS' if e > 400 else 'FAIL'}")

    # posterior predictive: within-item sd
    flat = [d for c in ch for d in c]
    by_item = {}
    for n in range(len(y)):
        by_item.setdefault(item[n], []).append(y[n])

    def sd(v):
        m = sum(v) / len(v)
        return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))

    obs_stat = sum(sd(v) for v in by_item.values()) / len(by_item)
    random.seed(99)
    reps = []
    for d in random.sample(flat, 500):
        sim = {}
        # The framing main effect is part of within-item spread, so it must be
        # simulated too; omitting it understates the replicated statistic.
        fj = [random.gauss(0, d["s_fram"]) for _ in range(n_fram)]
        for i in range(n_items):
            ai = random.gauss(0, d["s_item"])
            for j in range(n_fram):
                gij = random.gauss(
                    0, d["s_para"] if is_para[j] else d["s_null"])
                for _ in range(2):
                    sim.setdefault(i, []).append(
                        d["mu"] + ai + fj[j] + gij
                        + random.gauss(0, d["sigma_y"]))
        reps.append(sum(sd(v) for v in sim.values()) / n_items)
    reps.sort()
    print(f"\nposterior predictive (mean within-item sd):")
    print(f"  observed   {obs_stat:.4f}")
    print(f"  replicated 5%={reps[int(.05*len(reps))]:.4f} "
          f"50%={reps[len(reps)//2]:.4f} "
          f"95%={reps[int(.95*len(reps))]:.4f}")
    print(f"  ppp = {sum(1 for x in reps if x >= obs_stat)/len(reps):.3f}")

    # Is paraphrase sensitivity homogeneous across items? A Normal g assumes
    # it is. If the per-item spread is strongly skewed, that assumption is
    # wrong and the model will over-predict the typical item while
    # under-predicting the extreme one.
    print("\n=== per-item paraphrase spread (is a Normal g the right shape?) "
          "===")
    per_item = []
    for i in range(n_items):
        vals = [y[n] for n in range(len(y)) if item[n] == i and is_para[fram[n]]]
        per_item.append(max(vals) - min(vals))
    per_item.sort()
    med = per_item[len(per_item) // 2]
    print("  sorted per-item paraphrase ranges:")
    print("   ", " ".join(f"{v:.3f}" for v in per_item))
    print(f"  median {med:.4f}  max {per_item[-1]:.4f}  "
          f"max/median {per_item[-1]/max(med,1e-9):.1f}x")

    # prior sensitivity
    print("\n=== prior sensitivity (prereg section 6) ===")
    print(f"{'prior_scale':12} {'s_item':>9} {'s_null':>9} {'s_para':>9} "
          f"{'G':>9}")
    for ps in (0.5, 1.0, 2.0):
        c2 = fit(y, item, fram, is_para, n_items, n_fram, prior_scale=ps,
                 iters=3000, burn=1500)
        f2 = [d for c in c2 for d in c]
        row = [sum(d[p] for d in f2) / len(f2)
               for p in ("s_item", "s_null", "s_para", "G")]
        print(f"{ps:12.1f} {row[0]:9.5f} {row[1]:9.5f} {row[2]:9.5f} "
              f"{row[3]:9.5f}")


if __name__ == "__main__":
    main()
