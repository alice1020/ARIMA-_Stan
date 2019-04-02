data {
  int<lower=3> T; // num observations
  vector[T] y; // observed outputs
#int<lower=0> T_new;
  real y_t1_new;
  real y_t2_new;
}


parameters {
  real mu; // mean coeff
  // vector[K] b;  // population-level effects 
  real<lower=-1,upper=1> phi1;
  real<lower=-1,upper=1> phi2;
  //real deltavarphi; // moving avg coeff e_{t-13}
  real<lower=0> sigma; // noise scale
  vector[T] y_hat; // fitted values
  real y_new; // predicted values
}

model {    
  
  vector[T] nu; // prediction for time t
  vector[T] err; // error for time t
  nu[1] = mu + phi1 * mu + phi2 * mu; // assume err[0] == 0
  err[1] = y[1] - nu[1];
  for (t in 3:T) {
    nu[t] = mu + phi1 * y[t-1]  + phi2 * y[t-2];
    err[t] = y[t] - nu[t];
  }
  
  
  
  mu ~ normal(9,6); // priors
  phi1 ~ normal(0,6);
  phi2 ~ normal(0,6);
  sigma ~ cauchy(0,2); 
  //err ~ normal(0, sigma); // likelihood
  for (t in 3:T) {
    y[t] ~ normal(nu[t], sigma); // likelihood
  }
  y_new ~ normal(mu + phi1 * y_t1_new + phi2 * y_t2_new, sigma);// likelihood
}


generated quantities {
  vector[T] log_lik;
  for (t in 1:T){
    log_lik[t] = normal_rng(mu + phi1 * y_t1_new + phi2 * y_t2_new, sigma);
  }
}
