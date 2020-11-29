// "hierMD_MLT.stan" by Michael L. Thompson
// Copyright (C) 2020, Michael L. Thompson
// CONTACT VIA: https://www.linkedin.com/in/mlthomps
//
// Adapted from the Stan code in R script "VB_stan_hierMD.R", 
// which included the following:
//    IPG - IDSIA
//    Authors: L. Azzimonti, G. Corani
//    Reference: "Hierarchical estimation of parameters in Bayesian networks",
//
// Bibtex citation:
// @article{AZZIMONTI201967,
//   title = "Hierarchical estimation of parameters in Bayesian networks",
//   journal = "Computational Statistics & Data Analysis",
//   volume = "137",
//   pages = "67 - 91",
//   year = "2019",
//   issn = "0167-9473",
//   doi = "10.1016/j.csda.2019.02.004",
//   url = "http://www.sciencedirect.com/science/article/pii/S0167947319300519",
//   author = "Laura Azzimonti and Giorgio Corani and Marco Zaffalon"
// }
//
// Per personal communication with L. Azzimonti, the original IDSIA code was
// released under the GPLv3 license.  Therefore, this specific Stan program is 
// also released under the GPLv3 license:
//
//   This program is free software: you can redistribute it and/or modify
//   it under the terms of the GNU General Public License as published by
//   the Free Software Foundation, either version 3 of the License, or
//   (at your option) any later version.
// 
//   This program is distributed in the hope that it will be useful,
//   but WITHOUT ANY WARRANTY; without even the implied warranty of
//   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//   GNU General Public License for more details.
// 
//   You should have received a copy of the GNU General Public License
//   along with this program.  If not, see <https://www.gnu.org/licenses/>.
//
data {
  int<lower=2> n_st_ch; // number of child states
  int<lower=2> n_st_pr; // number of total combos of parent states
  int<lower=0> N_ch_pr[n_st_ch,n_st_pr]; // number of cases at all combos of parents & child
  vector<lower=0>[n_st_ch] alpha_0; // hyperparameter for Dirichlet priors
}

parameters {
  simplex[n_st_ch] theta[n_st_pr]; // conditional probability table parameters
  simplex[n_st_ch] alpha_norm; // population-level parameter for Dirichlet priors, normalized
  real<lower=0> N_prior; // number of cases represented by prior
}

transformed parameters {
  vector<lower=0>[n_st_ch] alpha; // population-level for Dirichlet priors
  alpha = N_prior * alpha_norm;
}

model {
  alpha_norm ~ dirichlet(alpha_0); // prior
  N_prior    ~ student_t(4,1,1);   // prior
  
  for (i_st_pr in 1:n_st_pr){
    theta[i_st_pr]    ~ dirichlet( alpha ); // prior
    if( sum(N_ch_pr[,i_st_pr]) > 0){ // (not really necessary...)
      N_ch_pr[,i_st_pr] ~ multinomial( theta[i_st_pr] );  // likelihood
    }
  }
}
