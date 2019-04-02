
//ARIMA(0,1,1)(1,0,1)[12]                    

data {
  int<lower=1> N;
  real y[N];
}

parameters {
  real trend[N];
  real season[N];
  real s_trend;
  real s_q;
  real s_season;
  real d;
  real mu;              // mean term
  real phi;             // autoregression coeff
  real theta;           // moving avg coeff
  real<lower=0> sigma;  // noise scale
}


model {
  real q[N];
  real cum_trend[N];
  vector[N] err;        // error for time t
  for (i in 12:N) {
    season[i]~normal(-season[i-1]-season[i-2]-season[i-3]-season[i-4]-season[i-5]-season[i-6]-season[i-7]-season[i-8]-season[i-9]-season[i-10]-season[i-11],s_season);
  }
  for (i in 3:N)
    trend[i]~normal(2*trend[i-1]-trend[i-2],s_trend);
  cum_trend[1]<-trend[1];
  for (i in 2:N)
    cum_trend[i]<-cum_trend[i-1]+trend[i];
  for (i in 1:N)
    q[i]<-y[i]-cum_trend[i]-season[i];
  for (i in 1:N)
    q[1] <- (mu + phi * mu + d, s_q);
  err[1] <- y[1] - q[1];
  for (i in 2:N) {
    q[i] <- mu + phi * y[i-1] + theta * err[i-1]  + d, s_q;
    err[i] <- y[i] - nu[i];
    
  }
  
  // priors
  mu ~ normal(0,10);
  phi ~ normal(0,2);
  theta ~ normal(0,2);
  sigma ~ cauchy(0,5);
  
  // likelihood
  err ~ normal(0,sigma);
}
