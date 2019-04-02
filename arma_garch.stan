data {
  int<lower=1> T;
  real y[T];
}
parameters {
  real y0;
  real<lower=0> sigma0;
  real con;
  real<lower=-1,upper=1> phi;
  real<lower=-1,upper=1> theta;
  simplex[3] alpha; 
}
transformed parameters {
  vector[T] err;
  vector<lower=0.0>[T] sigma;
  
  err[1] <- y[1] -  (con + phi * y0);
  sigma[1] <- sigma0;
  for (t in 2:T) {
    sigma[t] <- sqrt(alpha[1]
                       + alpha[2] * square(err[t-1])
                       + alpha[3] * square(sigma[t-1]));
                       err[t] <- y[t] - (con + phi * y[t-1] + theta * err[t-1]);
  }
}
model {
  // ARMA
  con ~ normal(9,1);
  //  phi ~ normal(0,2)T[-1,1];
  //  theta ~ normal(0,2)T[-1,1];
  // start values
  sigma0 ~ cauchy(0,2); // expected uncond. variance of ts is 1.0
  y0 ~ normal(con/(1-phi),sigma0/sqrt(1-square(phi))); // https://en.wikipedia.org/wiki/Autoregressive_model
  // likelihood
  err ~ normal(0,sigma);
}
generated quantities {
  real forecast;
  real ymse;
  // mean and Var forecast
  forecast <-  con + phi * y[T] + theta * err[T];
  ymse <- sqrt(alpha[1]
                 + alpha[2] * square(err[T])
                 + alpha[3] * square(sigma[T]));
}
