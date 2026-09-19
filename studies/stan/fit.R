# Bayesian fit of the nested measurement model (prereg sections 5-6, as amended
# by deviations D6/D7).
#
# Runs, in order:
#   1. prior predictive check   (before seeing the likelihood)
#   2. the fit                  (4 chains x 2000, per prereg)
#   3. convergence gate         (Rhat < 1.01, ESS_bulk > 400 for ALL parameters)
#   4. posterior predictive     (does it reproduce within-item spread?)
#   5. prior sensitivity        (half-normal 0.5 / 1 / 2)

suppressPackageStartupMessages(library(rstan))
options(mc.cores = 4)
rstan_options(auto_write = TRUE)

d <- read.csv("studies/stan/data.csv")
is_para <- tapply(d$is_para, d$framing, function(x) x[1])
is_para <- as.integer(is_para[order(as.integer(names(is_para)))])

stan_data <- list(
  N = nrow(d), I = max(d$item), F = max(d$framing),
  item = d$item, fram = d$framing, is_para = is_para,
  y = d$score, prior_scale = 1
)

cat("N =", stan_data$N, " I =", stan_data$I, " F =", stan_data$F, "\n")
cat("is_para =", is_para, "\n\n")

sm <- stan_model("studies/stan/model.stan")

# ---- 1. prior predictive -------------------------------------------------
cat("=== prior predictive check ===\n")
# The generative prior, drawn directly: every term is conjugate-free and
# independent, so simulating it needs no sampler.
set.seed(1)
n <- 20000
ps <- stan_data$prior_scale
prior_y <- rnorm(n, 1.5, 1) +                         # mu
  rnorm(n, 0, abs(rnorm(n, 0, ps))) +                 # a: item
  rnorm(n, 0, abs(rnorm(n, 0, ps))) +                 # f: framing main
  rnorm(n, 0, abs(rnorm(n, 0, ps))) +                 # g: interaction
  rnorm(n, 0, abs(rnorm(n, 0, ps)))                   # residual
cat(sprintf("implied prior y: 5%%=%.2f  50%%=%.2f  95%%=%.2f\n",
            quantile(prior_y, .05), median(prior_y), quantile(prior_y, .95)))
cat(sprintf("fraction of prior mass inside the valid 0-3 scale: %.2f\n",
            mean(prior_y >= 0 & prior_y <= 3)))
cat(sprintf("observed y range: %.3f .. %.3f\n\n", min(d$score), max(d$score)))

# ---- 2. fit --------------------------------------------------------------
cat("=== fitting ===\n")
fit <- sampling(sm, data = stan_data, chains = 4, iter = 2000,
                refresh = 0, seed = 20260919,
                control = list(adapt_delta = 0.99, max_treedepth = 15))

pars <- c("mu", "s_item", "s_fram", "s_null", "s_para", "sigma_y",
          "drift", "G")
print(summary(fit, pars = pars, probs = c(0.025, 0.5, 0.975))$summary)

# ---- 3. convergence gate -------------------------------------------------
cat("\n=== convergence gate (prereg section 6) ===\n")
s_all <- summary(fit)$summary
s_all <- s_all[!grepl("^y_rep", rownames(s_all)), , drop = FALSE]
s_all <- s_all[is.finite(s_all[, "Rhat"]), , drop = FALSE]
max_rhat <- max(s_all[, "Rhat"])
min_ess <- min(s_all[, "n_eff"])
nd <- sum(sapply(get_sampler_params(fit, inc_warmup = FALSE),
                 function(x) sum(x[, "divergent__"])))
cat(sprintf("max Rhat = %.4f (threshold < 1.01)  -> %s\n", max_rhat,
            ifelse(max_rhat < 1.01, "PASS", "FAIL")))
cat(sprintf("min ESS  = %.0f  (threshold > 400)   -> %s\n", min_ess,
            ifelse(min_ess > 400, "PASS", "FAIL")))
cat(sprintf("divergent transitions = %d\n", nd))

# ---- 4. posterior predictive --------------------------------------------
cat("\n=== posterior predictive: within-item sd ===\n")
yrep <- extract(fit, "y_rep")$y_rep
obs_stat <- mean(tapply(d$score, d$item, sd))
rep_stat <- apply(yrep, 1, function(r) mean(tapply(r, d$item, sd)))
cat(sprintf("observed mean within-item sd = %.4f\n", obs_stat))
cat(sprintf("replicated  5%%=%.4f 50%%=%.4f 95%%=%.4f\n",
            quantile(rep_stat, .05), median(rep_stat),
            quantile(rep_stat, .95)))
cat(sprintf("posterior predictive p = %.3f\n", mean(rep_stat >= obs_stat)))

# ---- 5. prior sensitivity ------------------------------------------------
cat("\n=== prior sensitivity (prereg section 6) ===\n")
cat(sprintf("%-14s %10s %10s %10s %10s\n",
            "prior_scale", "s_item", "s_null", "s_para", "G"))
for (ps in c(0.5, 1, 2)) {
  sd2 <- stan_data; sd2$prior_scale <- ps
  f2 <- sampling(sm, data = sd2, chains = 4, iter = 2000, refresh = 0,
                 seed = 20260919,
                 control = list(adapt_delta = 0.99, max_treedepth = 15))
  m <- summary(f2, pars = c("s_item", "s_null", "s_para", "G"))$summary[, "mean"]
  cat(sprintf("%-14.1f %10.4f %10.4f %10.4f %10.4f\n",
              ps, m["s_item"], m["s_null"], m["s_para"], m["G"]))
}

saveRDS(fit, "studies/stan/fit.rds")
cat("\nsaved studies/stan/fit.rds\n")
