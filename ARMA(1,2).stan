data {
  int<lower=3> T; // num observations
  vector[T] y; // observed outputs
#int<lower=0> T_new;
  real y_t1_new;
}


parameters {
  real mu; // mean coeff
  // vector[K] b;  // population-level effects 
  real<lower=-1,upper=1> phi1;
  real<lower=-1,upper=1> delta1; // moving avg coeff e_{t-1}
  real<lower=-1,upper=1> delta2; // moving avg coeff e_{t-2}
  //real deltavarphi; // moving avg coeff e_{t-13}
  real<lower=0> sigma; // noise scale
  vector[T] y_hat; // fitted values
  real y_new; // predicted values
}

model {    
  
  vector[T] nu; // prediction for time t
  vector[T] err; // error for time t
  nu[1] = mu + phi1 * mu; // assume err[0] == 0
  err[1] = y[1] - nu[1];
  nu[2] = mu + phi1 * mu; // assume err[0] == 0
  err[2] = y[2] - nu[2] - delta1 * err[1];
  nu[3] = mu + phi1 * mu; 
  err[3] = y[3] - nu[3] - delta1 * err[2] - delta2 * err[1];
  for (t in 3:T) {
    nu[t] = mu + phi1 * y[t-1] + delta1 * err[t-1]  + delta2 * err[t-2];
    err[t] = y[t] - nu[t];
  }
  
  
  
  mu ~ normal(9,6); // priors
  phi1 ~ normal(0,6);
  delta1 ~ normal(0,6);
  delta2 ~ normal(0,6);
  sigma ~ cauchy(0,2); 
  //err ~ normal(0, sigma); // likelihood
  for (t in 3:T) {
    y[t] ~ normal(nu[t], sigma); // likelihood
  }
  y_new ~ normal(mu + phi1 * y_t1_new, sigma);// likelihood
}





//generated quantities {
//  vector[T_new] y_hat;
//for (t in 1:T_new){
//y_hat[t] = normal_rng(mu + phi1 * y_new[t] + phi2 * y_new[t-1], sigma);
//}
//}
