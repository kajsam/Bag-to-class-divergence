function BH_KL = simulations_kernel_bag_to_bag(X, imp_bw, bw)

x_neg       = X{1};         % negative bags in training set
n_neg       = size(x_neg,1);
x_pos       = X{2};         % positive bags in training set
n_pos       = size(x_pos,1);
x_test      = X{3};         % test set
n_test_neg  = X{4};

%% Estimation of distributions

% Each  distribution will be approximated by a kernel density estimation.  

n_test = size(x_test,1);

%% Approximations of integrals by importance sampling
  
% Classification by minimum BH distance and KL information
BH_class = ones(1,n_test); % All positive class
KL_class = ones(1,n_test);
for j = 1: n_test  
  % The importance sample consists of all instances in the training set and
  % the instances in the bag up for classification. 
  x_imp = [x_neg; x_pos; x_test(j,:)]; 
  % The importance distribution is estimated. 
  [x_imp_sort,~] = sort(x_imp(:));
  [f_imp,~] = ksdensity(x_imp_sort,x_imp_sort, 'Kernel', 'epanechnikov','Bandwidth',imp_bw);
  f_imp(f_imp<=eps) = eps; % To avoid Inf
  n = length(f_imp);
    
  %% Kernel density estimation of bags

  % Kde for negative bags
  f_neg = zeros(n,n_neg); 
  for i = 1: n_neg
    [f_n,~] = ksdensity(x_neg(i,:),x_imp_sort, 'Kernel', 'epanechnikov','Bandwidth',bw);
    f_n(f_n<=eps) = eps; % To avoid Inf
    f_neg(:,i) = f_n;
  end

  % Kde for positive bags
  f_pos = zeros(n,n_pos);
  for i = 1: n_pos
    [f_p,~] = ksdensity(x_pos(i,:),x_imp_sort, 'Kernel', 'epanechnikov','Bandwidth',bw);
    f_p(f_p<=eps) = eps; % To avoid Inf
    f_pos(:,i) = f_p;
  end

  % Kde for the unlabelled bag
  [f_bag,~] = ksdensity(x_test(j,:),x_imp_sort, 'Kernel', 'epanechnikov','Bandwidth',bw);
  f_bag(f_bag<=eps) = eps;
  
  % Divergence from unlabelled bag to each positive bag in training set
  bag_BH = zeros(1,n_pos); 
  bag_KL = zeros(1,n_pos);
  for i = 1: n_pos
    % Calculate each term for the imp samp sum.
    bh_terms = sqrt(f_bag.*f_pos(:,i))./f_imp;
    kl_terms = (f_bag.*log(f_bag./f_pos(:,i)))./f_imp;

    bag_BH(i) = -log(sum(bh_terms)/n);
    bag_KL(i) = sum(kl_terms)/n;    
  end
  % Minimum divergence to positive bag
  min_pos_BH = min(bag_BH);
  min_pos_KL = min(bag_KL);
  
  % Divergence from unlabelled bag to each negative bag in training set
  for i = 1: n_neg
    % Calculate each term for the imp samp sum.
    bh_terms = sqrt(f_bag.*f_neg(:,i))./f_imp; 
    kl_terms = (f_bag.*log(f_bag./f_neg(:,i)))./f_imp; 

    bag_BH(i) = -log(sum(bh_terms)/n);
    bag_KL(i) = sum(kl_terms)/n;    
  end
  % Minimum divergence to negative bag
  min_neg_BH = min(bag_BH);
  
  % Label according to minimum divergence
  if min_neg_BH < min_pos_BH
    BH_class(j) = 0;
  end
  min_neg_KL = min(bag_KL);
  if min_neg_KL < min_pos_KL
    KL_class(j) = 0;
  end
end

% Calculate sensitivity and specificity
class_vec = zeros(1,n_test);
class_vec(n_test_neg+1:end) = 1;
  
BH = classperf(class_vec,BH_class,'Positive', 1, 'Negative', 0); 
SESP_BH = [BH.Sensitivity BH.Specificity];
  
KL = classperf(class_vec,KL_class,'Positive', 1, 'Negative', 0); 
SESP_KL = [KL.Sensitivity KL.Specificity];
      
BH_KL = [SESP_BH; SESP_KL];
    