require(rstan)
require(gplots)

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

load(url("https://www.cs.helsinki.fi/u/sakaya/tutorial/data/UML.RData"))
X1 <- GeneExpression.HL60 #Blood Cancer
X2 <- GeneExpression.MCF7 #Breast Cancer
dim(X1) #verify that the dimensions of the data are 78 x 1106
dim(X2) #verify that the dimensions of the data are 78 x 1106
X1[1:3,1:5] #examine a few values


cca <- "
	data {
		int<lower=0> N; // Number of samples
		int<lower=0> D; // Sum of the original dimensions of every view
		int<lower=0> M; // The number of views
		int<lower=0> K; // The latent dimension
		int<lower = 0> Dm[M]; //The original dimension for each view

		matrix[N, D] X; // The data matrix
	}

	parameters {
		matrix[N, K] Z; // The latent matrix
		matrix[K, D] W; // The weight matrix. You will be learning *one* single W with structured sparsity - both view-specific components and shared components are captured by this
		vector<lower=0>[M] tau; // View-specific noise terms
		matrix<lower=0>[M,K] alpha; // View-specific ARD prior
	}

	transformed parameters{
		vector<lower=0>[M] t_tau;
		matrix<lower=0>[M,K] t_alpha; 
 		t_alpha = inv(sqrt(alpha));
		t_tau = inv(sqrt(tau));
	}

	model {
		// You will need to loop through 1:Dm[m] for each view m. Increment ind seperately to index the concatenated X and W.
		int ind;
		tau ~ gamma(1,1);			
		to_vector(Z) ~ normal(0,1); // because sampling K dimensional standard normal multivariate is equivalent to sampling from k univariate standard normal distributions.
		to_vector(alpha) ~ gamma(1e-3,1e-3); // stack columns of alpha to a vector and sample quickly.	
		ind = 0;
		// There is a more efficient way to do this with ragged arrays, but the effort spent is hugely disproportionate to the speed-ups obtained.
		for (m in 1:M) {	
    		for (d in 1:Dm[m]) {
        		ind = ind + 1;      
        		W[,ind] ~ normal(0.0, t_alpha[m,]);
        		X[,ind] ~ normal(Z*W[,ind], t_tau[m]);  
       		}
		}
    }"

N <- 78
D <- c(1106,1106)# Data dimensions
K <- 5 					

data <- list(N = N, Dm = D, D=sum(D), M=2, K=K, X=cbind(X1,X2))
m <- stan_model(model_code = cca)
stan.fit.vb <- vb(m, data = data, algorithm = "meanfield",tol_rel_obj =1e-3, iter = 5000, grad_samples=1)

Z.vb <- apply(extract(stan.fit.vb,"Z")[[1]], c(2,3), mean)
alpha.vb <- apply(extract(stan.fit.vb,"alpha")[[1]], c(2,3), mean)
W.vb <- apply(extract(stan.fit.vb,"W")[[1]], c(2,3), mean)

heatmap.2(alpha.vb, col = bluered(100), dendrogram='none',trace='none', Rowv = FALSE, Colv = FALSE, key=FALSE)
alp=3
row = c(order(Z.vb[,alp], decreasing = TRUE)[1:5],order(Z.vb[,alp])[1:5])
col1 = order(W.vb[alp,1:D[1]], decreasing = TRUE)[1:20]
sub1 = X1[row,col1]
col2 = order(W.vb[alp, D[1]:sum(D)], decreasing = TRUE)[1:20]
sub2 = X1[row,col2]
heatmap.2(sub1, col = bluered(100),
          dendrogram='none',trace='none',
          Rowv = FALSE, Colv = FALSE, key=FALSE)
heatmap.2(sub2, col = bluered(100),
          dendrogram='none',trace='none',
          Rowv = FALSE, Colv = FALSE, key=FALSE)
