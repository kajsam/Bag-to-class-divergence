function distr = gaussian_mixture(K,x,EMparam)

rng('default')

maxiter = EMparam(1);
reps = EMparam(2);
reg  = EMparam(3);
if reps > 1
  start = 'randSample';
else
  start = 'plus';
end
options = statset('MaxIter',maxiter); 

fit_distr = cell(1,K);
for k = 1: K
  fit_distr{k} = fitgmdist(x,k,'Regularize',reg,'Options',options,'Replicates',reps, ...
                               'Start',start);
end
  
distr = fit_distr;