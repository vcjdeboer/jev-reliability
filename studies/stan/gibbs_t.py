"""Student-t interaction model — the fix the failed predictive check indicated.

The Normal model over-predicted how much a typical item moves (ppp 0.982): a
single common scale is stretched to cover two highly paraphrase-sensitive items,
inflating the predicted spread for the other twelve.

A Student-t interaction lets a few cells be far out without inflating the common
scale. Fitted as a scale mixture of normals (Andrews & Mallows 1974):

    g[i,j] | lambda ~ Normal(0, s_g^2 / lambda[i,j])
    lambda[i,j]     ~ Gamma(nu/2, nu/2)

which marginally gives g ~ t_nu(0, s_g). Every full conditional stays conjugate
except nu, which is slice-sampled under a Gamma(2, 0.1) prior — the standard
weakly-informative choice for degrees of freedom (Juarez & Steel 2010).

Fits BOTH models on the same data and compares them on the same posterior
predictive statistic, so the comparison is like for like.
"""

import csv
import math
import random
import sys

sys.path.insert(0, "studies/stan")
from gibbs import (  # noqa: E402
    PARAMS,
    ess,
    load,
    scale_logpost,
    slice_sample,
    split_rhat,
    summarise,
)


def lambda_logpost(lams, nu):
    """log p(lambda | nu) for lambda ~ Gamma(nu/2, nu/2), plus Gamma(2,0.1)."""
    if nu <= 2.01 or nu > 200:
        return -math.inf
    a = nu / 2.0
    n = len(lams)
    total = n * (a * math.log(a) - math.lgamma(a))
    for lam in lams:
        total += (a - 1.0) * math.log(lam) - a * lam
    # Gamma(2, 0.1) prior on nu
    total += math.log(nu) - 0.1 * nu
    return total


def run_chain_t(y, item, fram, is_para, n_items, n_fram, prior_scale,
                iters, burn, seed):
    random.seed(seed)
    N = len(y)

    cell_obs = {}
    for n in range(N):
        cell_obs.setdefault((item[n], fram[n]), []).append(n)
    item_obs, fram_obs = {}, {}
    for n in range(N):
        item_obs.setdefault(item[n], []).append(n)
        fram_obs.setdefault(fram[n], []).append(n)

    mu = sum(y) / N
    a = [0.0] * n_items
    f = [0.0] * n_fram
    g = {k: 0.0 for k in cell_obs}
    lam = {k: 1.0 for k in cell_obs}
    s_item = s_fram = s_null = s_para = sigma_y = 0.1
    nu = 6.0

    draws = []
    for it in range(iters):
        sy2 = sigma_y ** 2

        prec = 1.0 + N / sy2
        acc = 1.5
        for n in range(N):
            acc += (y[n] - a[item[n]] - f[fram[n]]
                    - g[(item[n], fram[n])]) / sy2
        mu = acc / prec + random.gauss(0, 1) / math.sqrt(prec)

        for i in range(n_items):
            idx = item_obs[i]
            prec = 1.0 / (s_item ** 2) + len(idx) / sy2
            acc = sum(y[n] - mu - f[fram[n]] - g[(item[n], fram[n])]
                      for n in idx) / sy2
            a[i] = acc / prec + random.gauss(0, 1) / math.sqrt(prec)

        for j in range(n_fram):
            idx = fram_obs[j]
            prec = 1.0 / (s_fram ** 2) + len(idx) / sy2
            acc = sum(y[n] - mu - a[item[n]] - g[(item[n], fram[n])]
                      for n in idx) / sy2
            f[j] = acc / prec + random.gauss(0, 1) / math.sqrt(prec)

        # interaction, now with a per-cell scale factor
        for (i, j), idx in cell_obs.items():
            s_g = s_para if is_para[j] else s_null
            prec = lam[(i, j)] / (s_g ** 2) + len(idx) / sy2
            acc = sum(y[n] - mu - a[i] - f[j] for n in idx) / sy2
            g[(i, j)] = acc / prec + random.gauss(0, 1) / math.sqrt(prec)

        # lambda | g, nu  ~ Gamma((nu+1)/2, (nu + g^2/s_g^2)/2)
        shape = (nu + 1.0) / 2.0
        for (i, j) in cell_obs:
            s_g = s_para if is_para[j] else s_null
            rate = (nu + (g[(i, j)] ** 2) / (s_g ** 2)) / 2.0
            lam[(i, j)] = random.gammavariate(shape, 1.0 / rate)

        # ridge-breaking joint shifts
        for vec, s_vec in ((a, s_item), (f, s_fram)):
            d = random.gauss(0, 0.05)
            logr = (-0.5 * ((mu + d - 1.5) ** 2 - (mu - 1.5) ** 2)
                    + sum(-0.5 * ((v - d) ** 2 - v * v) / (s_vec ** 2)
                          for v in vec))
            if math.log(random.random()) < logr:
                mu += d
                for k in range(len(vec)):
                    vec[k] -= d

        s_item = slice_sample(
            s_item, lambda s: scale_logpost(a, s, prior_scale))
        s_fram = slice_sample(
            s_fram, lambda s: scale_logpost(f, s, prior_scale))
        # standardised interactions: g * sqrt(lambda) is Normal(0, s_g)
        gn = [g[k] * math.sqrt(lam[k]) for k in g if not is_para[k[1]]]
        gp = [g[k] * math.sqrt(lam[k]) for k in g if is_para[k[1]]]
        s_null = slice_sample(
            s_null, lambda s: scale_logpost(gn, s, prior_scale))
        s_para = slice_sample(
            s_para, lambda s: scale_logpost(gp, s, prior_scale))
        resid = [y[n] - mu - a[item[n]] - f[fram[n]] - g[(item[n], fram[n])]
                 for n in range(N)]
        sigma_y = slice_sample(
            sigma_y, lambda s: scale_logpost(resid, s, prior_scale))

        lams = list(lam.values())
        nu = slice_sample(nu, lambda v: lambda_logpost(lams, v),
                          w=1.0, lower=2.02, upper=200.0)

        if it >= burn:
            draws.append({
                "mu": mu, "s_item": s_item, "s_fram": s_fram,
                "s_null": s_null, "s_para": s_para, "sigma_y": sigma_y,
                "nu": nu,
                "drift": s_para ** 2 - s_null ** 2,
                "G": s_item ** 2 / (s_item ** 2 + s_null ** 2 + sigma_y ** 2),
            })
    return draws


def sd(v):
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def ppc(draws, y, item, fram, is_para, n_items, n_fram, student_t, seed=99):
    """Replicate the mean within-item sd under the fitted model."""
    by_item = {}
    for n in range(len(y)):
        by_item.setdefault(item[n], []).append(y[n])
    observed = sum(sd(v) for v in by_item.values()) / len(by_item)

    random.seed(seed)
    reps = []
    for d in random.sample(draws, min(500, len(draws))):
        sim = {}
        fj = [random.gauss(0, d["s_fram"]) for _ in range(n_fram)]
        for i in range(n_items):
            ai = random.gauss(0, d["s_item"])
            for j in range(n_fram):
                s_g = d["s_para"] if is_para[j] else d["s_null"]
                if student_t:
                    nu = d["nu"]
                    lam = random.gammavariate(nu / 2.0, 2.0 / nu)
                    gij = random.gauss(0, s_g / math.sqrt(lam))
                else:
                    gij = random.gauss(0, s_g)
                for _ in range(2):
                    sim.setdefault(i, []).append(
                        d["mu"] + ai + fj[j] + gij
                        + random.gauss(0, d["sigma_y"]))
        reps.append(sum(sd(v) for v in sim.values()) / n_items)
    reps.sort()
    p = sum(1 for x in reps if x >= observed) / len(reps)
    return observed, reps, p


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "studies/stan/data-fixed.csv"
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
    burn = iters // 4
    y, item, fram, is_para, n_items, n_fram = load(path)
    print(f"data: {path}  N={len(y)} items={n_items} framings={n_fram}")
    print(f"chains=4 iters={iters} burn={burn}\n")

    print("=== Student-t interaction ===")
    chains = [run_chain_t(y, item, fram, is_para, n_items, n_fram, 1.0,
                          iters, burn, 2000 + k) for k in range(4)]
    r, e = summarise(chains, PARAMS + ["nu"])
    print(f"\nconvergence gate: max Rhat {r:.4f} "
          f"{'PASS' if r < 1.01 else 'FAIL'} | min ESS {e:.0f} "
          f"{'PASS' if e > 400 else 'FAIL'}")

    flat_t = [d for c in chains for d in c]
    obs, reps, p = ppc(flat_t, y, item, fram, is_para, n_items, n_fram, True)
    print("\nposterior predictive (mean within-item sd), Student-t:")
    print(f"  observed   {obs:.4f}")
    print(f"  replicated 5%={reps[int(.05*len(reps))]:.4f} "
          f"50%={reps[len(reps)//2]:.4f} 95%={reps[int(.95*len(reps))]:.4f}")
    print(f"  ppp = {p:.3f}")

    nus = sorted(d["nu"] for d in flat_t)
    print(f"\nnu: mean {sum(nus)/len(nus):.2f}  "
          f"95% CI [{nus[int(.025*len(nus))]:.2f}, "
          f"{nus[int(.975*len(nus))]:.2f}]")
    print("  (low nu = heavy tails = item-specific paraphrase sensitivity;"
          " nu -> 30+ is effectively Normal)")


if __name__ == "__main__":
    main()
