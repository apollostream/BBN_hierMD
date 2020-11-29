// "hierMDmix_MLT.stan" by Michael L. Thompson
// Copyright (C) 2020, Michael L. Thompson
// CONTACT VIA: https://www.linkedin.com/in/mlthomps
//
// Inspired by hierarchical Multinomial Dirichlet approach of Azzimonti & Corani
// in their R script "VB_stan_hierMD.R", which included the following:
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
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
// 
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
// 
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <https://www.gnu.org/licenses/>.
//
data {
  int<lower=2> n_st_ch; // number of child states
  int<lower=2> n_st_pr; // number of total combos of parent states
  int<lower=0> N_ch_pr[n_st_ch,n_st_pr]; // number of cases at all combos of parents & child
  vector<lower=0>[n_st_ch] alpha_0; // hyperparameter for Dirichlet priors
  
  int <lower=1> n_parent; // number of parents
  int <lower=2> n_st_pr_i[n_parent]; // number of states for each parent
  int i_st_pr[n_st_pr,n_parent]; // state of each parent for each combo of parents
      // in R: > i_st_pr <- as.matrix(expand.grid(purrr::map(n_st_pr_i,~seq(1,.x))))
}

transformed data {
  int n_st_pr_sum; // sum of number of states for parents
  vector[n_parent] alpha_pr_mix; // Dirichlet parameters for mixture
  
  n_st_pr_sum  = sum(n_st_pr_i);
  alpha_pr_mix = rep_vector(1,n_parent);
}

parameters {
  simplex[n_st_ch] theta_i[n_st_pr_sum];  // conditional probability table parameters
  simplex[n_st_ch] alpha_norm; // population-level parameter for Dirichlet priors, normalized
  real<lower=0> N_prior;  // number of cases represented by prior
  simplex[n_parent] w_mix; // mixture probabilities on the parents alpha_i
}

transformed parameters {
  vector<lower=0>[n_st_ch] theta[n_st_pr]; // mixture alpha
  vector<lower=0>[n_st_ch] alpha; // population level
  
  // unnormalized hyperparameters
  alpha = N_prior * alpha_norm;  // population-level

  // Mix conditional probability vector theta over each parent's state
  for( i_st in 1:n_st_pr ){
    theta[i_st] = rep_vector(0,n_st_ch); // initialize as zeros
    for( i in 1:n_parent ){
      // cummulative mixture contributions of parent states
      theta[i_st] += w_mix[i] * theta_i[ sum(head(n_st_pr_i,i-1)) + i_st_pr[i_st,i] ];
    }
  }
}

model {
  
  alpha_norm ~ dirichlet( alpha_0 );      // prior
  N_prior    ~ student_t( 4, 1, 1 );      // prior
  w_mix      ~ dirichlet( alpha_pr_mix ); // prior
  
  // sample alpha hyperparameter for each state of each parent
  {
    int i_st = 0;
    for( i in 1:n_parent ){
      for( j in 1:n_st_pr_i[i]){
        i_st += 1;
        theta_i[i_st] ~ dirichlet( alpha ); // prior
      }
    }
  }

  for (i_st in 1:n_st_pr){
    //theta[i_st]    ~ dirichlet( alpha_ch[i_st] ); // prior
    if( sum(N_ch_pr[,i_st]) > 0){ // (not really necessary...)
      N_ch_pr[,i_st] ~ multinomial( theta[i_st] );  // likelihood
    }
  }
}
