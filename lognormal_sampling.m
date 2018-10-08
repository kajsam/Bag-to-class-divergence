function X = lognormal_sampling(N, param_n, param_p, p_neg)
 
%% Set parameters

% Number of instances and bags
n_x   = N(1); % number of instances in each bag
n_neg = N(2); % number of negative bags of size 1xn_x 
n_pos = N(3); % number of positive bags of size 1xn_x
n_new = N(4); % number of new, unclassified bags. 
              

% Distribution parameters
% Mu_n        - negative mean, Gaussian distributed
mu_mu_n       = param_n(1);
sigma_mu_n    = param_n(2);
% Sigma_n     - negative variance, absolute of Gaussian distributed
mu_sigma_n    = param_n(3);
sigma_sigma_n = param_n(4);

% Mu_p        - positive mean, Gaussian distributed
mu_mu_p       = param_p(1);
sigma_mu_p    = param_p(2);
% Sigma_p     - positive variance, absolute of Gaussian distributed
mu_sigma_p    = param_p(3);
sigma_sigma_p = param_p(4);


%% Start sampling

% Let the ratio of positive to negative bags be the same as in the training
% sample.
n_new_neg = floor(n_new/2); %random('Binomial',n_new,n_neg/(n_pos+n_neg)); 
n_new_pos = n_new - n_new_neg;

% The sampling procedure for positive bags is the same as in samling.m, but
% for the negative bags, all instances in the same bag come from the same
% distribution. 

% Negative bags
n_x_p = random('Binomial',n_x,p_neg,[1 n_neg]);
n_x_n = n_x-n_x_p;

mu_p = random('Normal',mu_mu_p,sigma_mu_p,[1 n_neg]);
sigma_p = abs(random('Normal',mu_sigma_p,sigma_sigma_p,[1 n_neg]));
mu_n = random('Normal',mu_mu_n,sigma_mu_n,[1 n_neg]);
sigma_n = abs(random('Normal',mu_sigma_n,sigma_sigma_n,[1 n_neg]));

x_neg = zeros(n_neg,n_x); % Matrix of negative bags
for i = 1: n_neg 
  x_p = random('Normal',mu_p(i),sigma_p(i),[1 n_x_p(i)]);  
  x_n = random('Normal',mu_n(i),sigma_n(i),[1 n_x_n(i)]);
    
  x_neg(i,:) = [x_p x_n];
end
X{1} = x_neg;

% New negative bags

n_x_p = random('Binomial',n_x,p_neg,[1 n_new_neg]);
n_x_n = n_x-n_x_p;

mu_p = random('Normal',mu_mu_p,sigma_mu_p,[1 n_new_neg]);
sigma_p = abs(random('Normal',mu_sigma_p,sigma_sigma_p,[1 n_new_neg]));
mu_n = random('Normal',mu_mu_n,sigma_mu_n,[1 n_new_neg]);
sigma_n = abs(random('Normal',mu_sigma_n,sigma_sigma_n,[1 n_new_neg]));

x_new_neg = zeros(n_new_neg,n_x); % Matrix of negative bags
for i = 1:n_new_neg
  x_p = random('Normal',mu_p(i),sigma_p(i),[1 n_x_p(i)]);  
  x_n = random('Normal',mu_n(i),sigma_n(i),[1 n_x_n(i)]);
    
  x_new_neg(i,:) = [x_p x_n];
end

% Positive bags in the training set

mu_mu_log = log(10);
sigma_mu_log =  sigma_mu_n*0.2; % No variance for sim 5
mu_sigma_log = sqrt(0.04);
sigma_sigma_log = sigma_mu_n*0.2;

mu_log = random('Normal',mu_mu_log,sigma_mu_log,[1 n_pos]);
sigma_log = abs(random('Normal',mu_sigma_log,sigma_sigma_log,[1 n_pos]));

x_pos = zeros(n_pos,n_x); % Matrix of positive bags
for i = 1: n_pos
  x_pos_p = random('Lognormal',mu_log(i),sigma_log(i),[1 n_x]);
  x_pos(i,:) = x_pos_p;
end
X{2} = x_pos;

% New positive bags

x_new_pos = zeros(n_new_pos,n_x); % Matrix of positive bags
mu_log = random('Normal',mu_mu_log,sigma_mu_log,[1 n_new_pos]);
sigma_log = abs(random('Normal',mu_sigma_log,sigma_sigma_log,[1 n_new_pos]));

for i = 1:n_new_pos
  x_pos_p = random('Lognormal',mu_log(i),sigma_log(i),[1 n_x]);    
  x_new_pos(i,:) = x_pos_p;
end

x_new = [x_new_neg; x_new_pos];
X{3} = x_new;
X{4} = n_new_neg;
