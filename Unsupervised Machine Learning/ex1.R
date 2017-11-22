require(rstan)
#example
gaussian <- "
	data {
	    int<lower=1> n;
	    vector[n] x;
	}
	parameters {
    	    real mu;
            real<lower=0> sigma;
	}
	model {
	    mu ~ normal(0, 10);
	    x ~ normal(mu, sigma);
   	} "


n <- 1000
x <- rnorm(n, 5, 10)
data <- list(x=x, n=n)
m <- stan_model(model_code = gaussian)
samples <- sampling(m, data=data, iter=10000, chains=1)
mu <- mean(extract(samples)$mu)
sigma <- mean(extract(samples)$sigma)

#q1
require(rstan)
bernoulli <- "
	data { 
		int<lower=1> N;
		int<lower=0,upper=1> y[N];
	} 
	parameters {
		real theta;
	} 
	model {
		theta ~ beta(1,1);
		y ~ bernoulli(theta);
	}"

N <- 100
y <- rbinom(N, 1, .4)
data <- list(y=y, N=N)
m <- stan_model(model_code = bernoulli)
samples <- sampling(m, data=data, iter=1000, chains=1)
mu <- mean(extract(samples)$theta)

#q2
require(rstan)
binomial <- "
	data {
		int<lower=1> N;
		int y[N];
	}
	parameters {
		real theta;
	}
	model { 
		theta ~ beta(1,1);
		y ~ binomial(N,theta);
	}"

N <- 100
s <- 10 #number of trials
theta <- .4
y <- rbinom(N, 100, theta)
data <- list(y=y, N=N, s = s)
m <- stan_model(model_code = binomial)
samples <- sampling(m, data=data, iter=1000, chains=1)
theta <- mean(extract(samples)$theta)

#q3
require(rstan)
poisson <- "
	data {
		int<lower=1> N;
		int y[N];
	}
	parameters {
		real lambda;
	}
	model { 
		lambda ~ gamma(1,1);
		y ~ poisson(lambda);
	}"

N <- 100
lambda <- 10
y <- rpois(N,lambda)
data <- list(y=y, N=N)
m <- stan_model(model_code = poisson)
samples <- sampling(m, data=data, iter=1000, chains=1)
lambda <- mean(extract(samples)$lambda)

#q4
require(rstan)
normal <- "
	data {
		int<lower=1> N;
		vector[N] y;
	}
	parameters {
    real mu;
	}
	model {
    mu ~ normal(0,1);
    y ~ normal(mu,5);
	}"
N<-100
mu <- 10
n <- 100
y <- rnorm(N, 10, 3)
data <- list(y=y, n=n)
m <- stan_model(model_code = normal)
samples <- sampling(m, data=data, iter=1000, chains=1)
mu <- mean(extract(samples)$mu)

#q5
normal <- "
	data {
    int<lower=1> N;
		vector[N] y;
  }
  parameters {
    real mu;
    real sigma2;
  }
  transformed parameters {
  }
  model {
    sigma2 ~ inv_gamma(0.1,  0.1);
    mu ~ normal(0,1);
    y ~ normal(mu,sigma2);
  }"

mu <- 10
sigma <- 5
N <- 100
y <- rnorm(N, mu, sigma)
data <- list(y=y, n=N)
m <- stan_model(model_code = normal)
samples <- sampling(m, data=data, iter=1000, chains=1)
mu <- mean(extract(samples)$mu)
sigma2 <- mean(extract(samples)$sigma2)

#q6
require(rstan)
require(MASS)

normal <- "
  data {
    int<lower=1> n;
    int<lower=1> D;
		matrix[n,D] y;
  }
  parameters {
    vector[n] mu;
    real sigma2;
  }
  transformed parameters {
  }
  model {
    sigma2 ~ inv_gamma(0.1,  0.1);
    for(i in 1:n){
      mu[i] ~ normal(0,sigma2);
      y[i] ~ normal(mu[i],sigma2);
    }
  }"

D <- 10
mu <- rnorm(D) + 10
sigma2 <- 5
n <- 100
y <- mvrnorm(n, mu , sigma2*diag(D))
data <- list(y=y, n=n, D=D)
m <- stan_model(model_code = normal)
samples <- sampling(m, data=data, iter=1000, chains=1)
mu <- colMeans(extract(samples)$mu)
sigma2 <- mean(extract(samples)$sigma2)

#q7
require(rstan)
bayesian_linear <- "
  data {
    int<lower=1> N;
    int<lower=1> D;
		matrix[N,D] X;
    vector[N] y;
  }
  parameters {
    real tau;
    vector[D] w;
    real<lower=0> sigma2;
  }
  transformed parameters {
  }
  model {
    tau ~ gamma(1,1);
    sigma2 ~ inv_gamma(1,1);
    for(i in 1:D){
      w[i] ~ normal(0,1/tau);
    }
    for(i in 1:N){
      y[i] ~ normal(dot_product(w',X[i,]),sigma2);
    }
  }"

tau <- 1
N <- 1000
D <- 10
w <- rnorm(D, sd = tau)
X <- matrix(rnorm(N*D), N, D)
y <- c(X %*% w + rnorm(N))
data <- list(N=N, D=D, X=X, y=y)

m <- stan_model(model_code = bayesian_linear)
samples <- sampling(m, data=data, iter=2000, chains=1)
w <- colMeans(extract(samples)$w)
tau <- mean(extract(samples)$tau)

#q8
require(rstan)
bayesian_linear_ard <- "
  data {
    int<lower=1> N;
    int<lower=1> D;
		matrix[N,D] X;
    vector[N] y;
  }
  parameters {
    vector[D] w;
    vector<lower=0>[D] alpha;
    real<lower=1> sigma2;
  }
  transformed parameters {
  }
  model {
    sigma2 ~ inv_gamma(1,1);
    for(i in 1:D){
      alpha[i] ~ gamma(0.01,0.01);
      w[i] ~ normal(0,1/alpha[i]);
    }
    for(i in 1:N){
      y[i] ~ normal(dot_product(w',X[i,]),sigma2);
    }
  }"
N <- 1000
D <- 10
alpha <- rep(1,D)
alpha[1:5] <- 1e6
w <- sapply(1/sqrt(alpha), function(a) rnorm(1,sd=a))
X <- matrix(rnorm(N*D), N, D)
y <- c(X %*% w + rnorm(N))
data <- list(N=N, D=D, X=X, y=y)

m <- stan_model(model_code = bayesian_linear_ard)
samples <- sampling(m, data=data, iter=2000, chains=1)
w <- colMeans(extract(samples)$w)
alpha <- colMeans(extract(samples)$alpha)

#q9
require(boot)

bayesian_logistic_ard <- "
  data {
    int<lower=1> N;
    int<lower=1> D;
    matrix[N,D] X;
    vector[N] y;
  }
  parameters {
    vector[D] w;
    vector[D] alpha;
  }
  transformed parameters {
  }
  model {
    sigma2 ~ inv_gamma(1,1);
    for(i in 1:D){
      alpha[i] ~ gamma(0.1,0.1);
      w[i] ~ normal(0,1/alpha[i]);
    }
    for(i in 1:N){
    y[i] ~ bern(sigma(dot_product(w',X[i,])));
    }
  }"


tau <- 1
N <- 1000
D <- 10
alpha <- rep(1,D)
alpha[1:5] <- 1e6
w <- sapply(1/sqrt(alpha), function(a) rnorm(1,sd=a))
X <- matrix(rnorm(N*D), N, D)
y <- rbinom(N,1, inv.logit(c(X %*% w + rnorm(N))))
data <- list(N=N, D=D, X=X, y=y)

m <- stan_model(model_code = bayesian_logistic_ard)
samples <- sampling(m, data=data, iter=2000, chains=1)
w <- colMeans(extract(samples)$w)
alpha <- colMeans(extract(samples)$alpha)

#q10
gmm <- "
	data {
    int<lower=1> N;
    int<lower=1> D;
    int<lower=1> K;
    matrix[N,D] X;
  }
  parameters {
    vector[K] sigma2;
    vector[K] mu;
    vector[K] p;
  }
  transformed parameters {
    int z[N];
    vector[K] a0;
    for(i in 1:K){
      a0[i] = 0.5;
    }
  }
  model {
    p ~ dirichlet(a0);
    for(i in 1:K){
      sigma2[i] ~ inv_gamma(1,1);
      mu[i] ~ normal(0,3);
    }
    for(i in 1:N){
      z[i] ~ categorical(p);
      x[i] ~ normal(mu[z[i]],sigma[z[i]]);
    }
  }"

N <- 1000
K <- 5
D <- 2
X <- matrix(rep(5 * rnorm(K * D),times = N/K), N, D, byrow=T) + rnorm(N*D)
data <- list(X=X, N=N, K=K, D=D)
m <- stan_model(model_code = gmm)
samples <- sampling(m, data=data, iter=1000, chains=1)