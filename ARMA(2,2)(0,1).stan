data {
  int<lower=15> T; // num observations
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
  real<lower=-1,upper=1> delta1; // moving avg coeff e_{t-1}
  real<lower=-1,upper=1> delta2; // moving avg coeff e_{t-2}
  real<lower=-1,upper=1> varphi; // moving avg coeff e_{t-12}
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
  nu[2] = mu + phi1 * mu + phi2 * mu; // assume err[0] == 0
  err[2] = y[2] - nu[2] - delta1 * err[1];
  nu[3] = mu + phi1 * mu + phi2 * mu; 
  err[3] = y[3] - nu[3] - delta1 * err[2] - delta2 * err[1];
  nu[4] = mu + phi1 * mu + phi2 * mu; 
  err[4] = y[4] - nu[4] - delta1 * err[3] - delta2 * err[2];
  nu[5] = mu + phi1 * mu + phi2 * mu; 
  err[5] = y[5] - nu[5] - delta1 * err[4] - delta2 * err[3];
  nu[6] = mu + phi1 * mu + phi2 * mu; 
  err[6] = y[6] - nu[6] - delta1 * err[5] - delta2 * err[4];
  nu[7] = mu + phi1 * mu + phi2 * mu;
  err[7] = y[7] - nu[7] - delta1 * err[6] - delta2 * err[5];
  nu[8] = mu + phi1 * mu + phi2 * mu; 
  err[8] = y[8] - nu[8] - delta1 * err[7] - delta2 * err[6];
  nu[9] = mu + phi1 * mu + phi2 * mu;
  err[9] = y[9] - nu[9] - delta1 * err[8] - delta2 * err[7];
  nu[10] = mu + phi1 * mu + phi2 * mu; 
  err[10] = y[10] - nu[10] - delta1 * err[9] - delta2 * err[8];
  nu[11] = mu + phi1 * mu + phi2 * mu;
  err[11] = y[11] - nu[11] - delta1 * err[10] - delta2 * err[9];
  nu[12] = mu + phi1 * mu + phi2 * mu; 
  err[12] = y[12] - nu[12] - delta1 * err[11] - delta2 * err[10];
  nu[13] = mu + phi1 * mu + phi2 * mu;
  err[13] = y[13] - nu[13] - delta1 * err[12] - delta2 * err[11] - varphi * err[1];
  nu[14] = mu + phi1 * mu + phi2 * mu;
  err[14] = y[14] - nu[14] - delta1 * err[13] - delta2 * err[12] - varphi * err[2] - delta1 * varphi * err[1];
  nu[15] = mu + phi1 * mu + phi2 * mu;
  err[15] = y[15] - nu[15] - delta1 * err[14] - delta2 * err[13] - varphi * err[3]  - delta1 * varphi * err[2] - delta2 * varphi * err[1];
  
  for (t in 15:T) {
    nu[t] = mu + phi1 * y[t-1] + phi2 * y[t-2] + delta1 * err[t-1]  + delta2 * err[t-2] + varphi * err[t-12] + delta1 * varphi * err[t-13] + delta2 * varphi * err[t-14];
    err[t] = y[t] - nu[t];
  }
  
  
  
  mu ~ normal(9,6); // priors
  phi1 ~ normal(0,6);
  phi2 ~ normal(0,6);
  delta1 ~ normal(0,6);
  delta2 ~ normal(0,6);
  varphi ~ normal(0,6);
  sigma ~ cauchy(0,2); 
  //err ~ normal(0, sigma); // likelihood
  for (t in 15:T) {
    y[t] ~ normal(nu[t], sigma); // likelihood
  }
  y_new ~ normal(mu + phi1 * y_t1_new + phi2 * y_t2_new, sigma);// likelihood
}





//generated quantities {
//  vector[T_new] y_hat;
//for (t in 1:T_new){
//y_hat[t] = normal_rng(mu + phi1 * y_new[t] + phi2 * y_new[t-1], sigma);
//}
//}
