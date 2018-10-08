function [param_n, param_p, flip, P] = simulation_param(k)

% Distribution parameters
% Mu_n        - negative mean, Gaussian distributed
mu_mu_n       = [0 0 0 0 9.5 9.5];
sigma_mu_n    = [sqrt(10) sqrt(10) sqrt(10) sqrt(10) 0 1];
% Sigma_n     - negative variance, absolute of Gaussian distributed
mu_sigma_n    = [1 1 1 1 sqrt(2.5) sqrt(2.5)];
sigma_sigma_n = [1 1 1 1 0 1];

flip          = [1 1 1 2 1 1]; 
% Mu_p        - positive mean, Gaussian distributed
mu_mu_p       = [15 15 0 15 13.5 13.5]; 
sigma_mu_p    = [sqrt(10) sqrt(10) sqrt(10) sqrt(10) 0 1];
% Sigma_p     - positive variance, absolute of Gaussian distributed
mu_sigma_p    = [1 1 10 1 sqrt(2.5) sqrt(2.5)];
sigma_sigma_p = [1 1 1 0 0 1];

% Instances sampled from positive distribution
p_neg         = [0 0.01 0 0.01 0.1 0.1]; % probability of positive sample in negative bag
p_pos         = [0.1 0.1 0.1 0.1 0.1 0.1]; % probability of positive sample in positive bag

flip = flip(k);
P = [p_neg(k) p_pos(k)];
param_n = [mu_mu_n(k) sigma_mu_n(k) mu_sigma_n(k) sigma_sigma_n(k)];
param_p = [mu_mu_p(k) sigma_mu_p(k) mu_sigma_p(k) sigma_sigma_p(k)];

