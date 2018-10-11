function [p_imp,y] = importance_sampling(x_samp,n_y)

% Importance sampling
y_samp = sort(x_samp(:));
% density estimate, from sample data, evaluated at sample data
p_samp = ksdensity(y_samp,y_samp);
% random sample of n_x data points from the x_samp population weighted by the p_samp density
y = randsample(y_samp,n_y,true,p_samp); 
y = sort(y);

p_imp = ksdensity(y,y); 