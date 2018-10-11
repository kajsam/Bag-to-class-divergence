function [rBH, rKL, cKL] = divergences_one(x_dbag, bag_class)

% Leave-one-out for divergence calculation

n_y = 1000; % Number of samples for the importance sampling

n_bag = length(bag_class);
D = size(x_dbag{1},2);

BH_neg = zeros(1,n_bag);
BH_pos = zeros(1,n_bag);
KL_neg = zeros(1,n_bag);
KL_pos = zeros(1,n_bag);
cKL = zeros(n_bag,D);
rBH = zeros(n_bag,D);
rKL = zeros(n_bag,D);
  
neg_idx = find(bag_class == 0);
pos_idx = find(bag_class == 1);

for d = 1: D
    
  % Create the negative and the positive class
  x_neg = [];
  for i = neg_idx
    x_neg = [x_neg; x_dbag{i}(:,d)];
  end 
  
  x_pos = [];
  for i = pos_idx
    x_pos = [x_pos; x_dbag{i}(:,d)];
  end 
 
  % Importance sampling
  x_samp = [x_neg; x_pos];
  [p_imp,y] = importance_sampling(x_samp,n_y);
      
  p_imp(p_imp<eps) = eps; % To avoid Inf
  n_y = length(p_imp);
  veps = eps*ones(n_y,1);
  
  % Density estimation of the classes
  
  p_neg = ksdensity(x_neg(:),y); 
  p_neg(p_neg<eps) = eps; 
  p_pos = ksdensity(x_pos(:),y); 
  p_pos(p_pos<eps) = eps;
  
  for j = 1: n_bag
    
    if bag_class(j) == 0
      % Kde for all negative instances except the bag
      x_neg = [];
      for i = setdiff(neg_idx,j)
        x_neg = [x_neg; x_dbag{j}(:,d)];
      end
      
      p_neg = ksdensity(x_neg(:),y); 
      p_neg(p_neg<eps) = eps; 
    else
      % Kde for all positive instances except the bag
      x_pos = [];
      for i = setdiff(pos_idx,j)
        x_pos = [x_pos; x_dbag{j}(:,d)];
      end   
      
      p_pos = ksdensity(x_pos(:),y); 
      p_pos(p_pos<eps) = eps;
    end
    
    % Kde for the bag      
    p_bag = ksdensity(x_dbag{j}(:,d),y); 
        
    % Calculate each term for the imp samp sum.
   
    bh_terms_neg = sqrt(p_bag.*p_neg)./p_imp+veps; 
    bh_terms_pos = sqrt(p_bag.*p_pos)./p_imp+veps;
    kl_terms_neg = p_bag.*log(p_bag./p_neg)./p_imp+veps;
    kl_terms_pos = p_bag.*log(p_bag./p_pos)./p_imp+veps;
    
    kl_terms_neg(p_bag < eps) = eps; % 0log0
    kl_terms_pos(p_bag < eps) = eps;
    
    w = p_pos./p_neg;
    cc_terms = w.*kl_terms_neg;
 
    BH_neg(j) = -log(sum(bh_terms_neg)/n_y);
    BH_pos(j) = -log(sum(bh_terms_pos)/n_y);
    KL_neg(j) = sum(kl_terms_neg)/n_y;    
    KL_pos(j) = sum(kl_terms_pos)/n_y;
    cKL(j,d) = sum(cc_terms)/n_y;
  end
 
  rBH(:,d) = BH_neg./BH_pos; % Classification by ratio
  rKL(:,d) = KL_neg./KL_pos;
end
        