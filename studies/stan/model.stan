// Nested measurement model for Jev replicate grids.
//
// OCRbayes structure (measurement cycle -> well -> plate) mapped onto
// repeat -> framing -> item, with the middle facet split so that the
// framing-induced shift has a separate scale for semantically-null
// perturbations and for paraphrases. Their contrast identifies construct drift.
//
// The quantity of interest is how much a framing moves A GIVEN ITEM, so the
// framing enters as an item x framing interaction, not only as a main effect.
//
// Non-centred throughout: the variance components are near zero on this scale
// and a centred parameterisation funnels badly.

data {
  int<lower=1> N;                              // observations
  int<lower=1> I;                              // items
  int<lower=1> F;                              // framings
  array[N] int<lower=1, upper=I> item;
  array[N] int<lower=1, upper=F> fram;
  array[F] int<lower=0, upper=1> is_para;
  vector[N] y;                                 // score, 0-3 scale
  real<lower=0> prior_scale;                   // half-normal sd, for sensitivity
}

parameters {
  real mu;
  vector[I] z_item;
  vector[F] z_fram;
  matrix[I, F] z_int;
  real<lower=0> s_item;                        // between items
  real<lower=0> s_fram;                        // framing main effect
  real<lower=0> s_null;                        // item x null-framing
  real<lower=0> s_para;                        // item x paraphrase
  real<lower=0> sigma_y;                       // repeat / residual
}

transformed parameters {
  vector[I] a = z_item * s_item;
  vector[F] f = z_fram * s_fram;
  matrix[I, F] g;
  for (j in 1 : F) {
    g[ : , j] = z_int[ : , j] * (is_para[j] == 1 ? s_para : s_null);
  }
}

model {
  vector[N] eta;
  for (n in 1 : N) {
    eta[n] = mu + a[item[n]] + f[fram[n]] + g[item[n], fram[n]];
  }

  mu ~ normal(1.5, 1);
  z_item ~ std_normal();
  z_fram ~ std_normal();
  to_vector(z_int) ~ std_normal();

  // lower=0 declarations make these half-normal
  s_item ~ normal(0, prior_scale);
  s_fram ~ normal(0, prior_scale);
  s_null ~ normal(0, prior_scale);
  s_para ~ normal(0, prior_scale);
  sigma_y ~ normal(0, prior_scale);

  y ~ normal(eta, sigma_y);
}

generated quantities {
  // RQ3: construct drift on the variance scale
  real drift = square(s_para) - square(s_null);
  // RQ4: reliability for this (unsaturated) region
  real G = square(s_item)
           / (square(s_item) + square(s_null) + square(sigma_y));
  vector[N] y_rep;
  for (n in 1 : N) {
    y_rep[n] = normal_rng(mu + a[item[n]] + f[fram[n]]
                          + g[item[n], fram[n]], sigma_y);
  }
}
