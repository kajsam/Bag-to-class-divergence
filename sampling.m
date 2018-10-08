function X = sampling(N, param_n, param_p, P, flip)

% Sampling bags and corresponding instances, both for training and test
% set. 

% Number of instances and bags
n_x   = N(1);   % number of instances in each bag
n_neg = N(2);   % number of negative bags of size 1xn_x 
n_pos = N(3);   % number of positive bags of size 1xn_x
n_test = N(4);  % number of bags in test set. 
              
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

% Instances sampled from positive distribution
p_neg         = P(1);   % in positive bags
p_pos         = P(2);   % in negative bags

%% Start sampling

% For the test set, half of the bags are negative/positive
n_test_neg = floor(n_test/2); % random('Binomial',n_test,0.5);
n_test_pos = ceil(n_test/2);

% Sample the parameters for each bag, then sample the instances from each 
% bag. 

% Training set - negative bags
x_neg = zeros(n_neg,n_x); % Matrix of negative bags

flipvec = random('Discrete Uniform',flip,[1 n_neg]);
if flip == 2 % if the positive instance means are either \mu or -\mu
  flipvec = ((-1).*ones(1,n_neg)).^flipvec;
end

% The fraction of positive instances in each bag is a sample from a 
% binomial distribution. 
n_x_p = random('Binomial',n_x,p_neg,[1 n_neg]);
n_x_n = n_x-n_x_p;

% The mean and variance for the positive and negative instances are 
% sampled from Gaussian distributions. 
mu_p = random('Normal',mu_mu_p,sigma_mu_p,[1 n_neg]);
mu_p = mu_p.*flipvec; % either \mu or -\mu
sigma_p = abs(random('Normal',mu_sigma_p,sigma_sigma_p,[1 n_neg]));
mu_n = random('Normal',mu_mu_n,sigma_mu_n,[1 n_neg]);
sigma_n = abs(random('Normal',mu_sigma_n,sigma_sigma_n,[1 n_neg]));

% Sample the positive and negative instances
for i = 1: n_neg  
  x_p = random('Normal',mu_p(i),sigma_p(i),[1 n_x_p(i)]);  
  x_n = random('Normal',mu_n(i),sigma_n(i),[1 n_x_n(i)]); 
  x_neg(i,:) = [x_p x_n];
end
X{1} = x_neg;

% Negative bags in the test set. Sampling procedure equivalent to training.
x_test_neg = zeros(n_test_neg,n_x); % Matrix of negative bags

flipvec = random('Discrete Uniform',flip,[1 n_test_neg]);
if flip == 2 
  flipvec = ((-1).*ones(1,n_test_neg)).^flipvec;
end

n_x_p = random('Binomial',n_x,p_neg,[1 n_test_neg]);
n_x_n = n_x-n_x_p;

mu_p = random('Normal',mu_mu_p,sigma_mu_p,[1 n_test_neg]);
mu_p = flipvec.*mu_p; 
sigma_p = abs(random('Normal',mu_sigma_p,sigma_sigma_p,[1 n_test_neg]));
mu_n = random('Normal',mu_mu_n,sigma_mu_n,[1 n_test_neg]);
sigma_n = abs(random('Normal',mu_sigma_n,sigma_sigma_n,[1 n_test_neg]));

for i = 1:n_test_neg
  x_p = random('Normal',mu_p(i),sigma_p(i),[1 n_x_p(i)]);  
  x_n = random('Normal',mu_n(i),sigma_n(i),[1 n_x_n(i)]);  
  x_test_neg(i,:) = [x_p x_n];
end

% The sampling procedure of positive bags is equivalent to that of negative
% bags. 

% Positive bags in the training set
x_pos = zeros(n_pos,n_x); % Matrix of bags

flipvec = random('Discrete Uniform',flip,[1 n_pos]);
if flip == 2 
  flipvec = ((-1).*ones(1,n_pos)).^flipvec;
end

n_x_p = random('Binomial',n_x,p_pos,[1 n_pos]);
n_x_n = n_x-n_x_p;

mu_p = random('Normal',mu_mu_p,sigma_mu_p,[1 n_pos]);
mu_p = flipvec.*mu_p;
sigma_p = abs(random('Normal',mu_sigma_p,sigma_sigma_p,[1 n_pos]));
mu_n = random('Normal',mu_mu_n,sigma_mu_n,[1 n_pos]);
sigma_n = abs(random('Normal',mu_sigma_n,sigma_sigma_n,[1 n_pos]));

for i = 1: n_pos  
  x_p = random('Normal',mu_p(i),sigma_p(i),[1 n_x_p(i)]);  
  x_n = random('Normal',mu_n(i),sigma_n(i),[1 n_x_n(i)]);  
  x_pos(i,:) = [x_p x_n];
end
X{2} = x_pos;

% Positive bags in the test set
x_test_pos = zeros(n_test_pos,n_x); % Matrix of bags

flipvec = random('Discrete Uniform',flip,[1 n_test_pos]);
if flip == 2 
  flipvec = ((-1).*ones(1,n_test_pos)).^flipvec;
end

n_x_p = random('Binomial',n_x,p_pos,[1 n_test_pos]);
n_x_n = n_x-n_x_p;

mu_p = random('Normal',mu_mu_p,sigma_mu_p,[1 n_test_pos]);
mu_p = flipvec.*mu_p;
sigma_p = abs(random('Normal',mu_sigma_p,sigma_sigma_p,[1 n_test_pos]));
mu_n = random('Normal',mu_mu_n,sigma_mu_n,[1 n_test_pos]);
sigma_n = abs(random('Normal',mu_sigma_n,sigma_sigma_n,[1 n_test_pos]));

for i = 1:n_test_pos
  x_p = random('Normal',mu_p(i),sigma_p(i),[1 n_x_p(i)]);  
  x_n = random('Normal',mu_n(i),sigma_n(i),[1 n_x_n(i)]);
  x_test_pos(i,:) = [x_p x_n];
end

x_test = [x_test_neg; x_test_pos];
X{3} = x_test;
X{4} = n_test_neg; % Need to know class labels in test set
