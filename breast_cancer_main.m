function breast_cancer_main(datafile)

% Requires: bag_identification.m, gaussian_mixture.m, 
%           bag_to_class_divergence.m, AUC_ROC.m

% Input:    datafile - string : The data can be found at 
%           https://figshare.com/articles/MIProblems_A_repository_of_
%           multiple_instance_learning_datasets/6633983

% "UCSB Breast is an image classification problem. The original datasets 
% consists of 58 TMA image excerpts (896 x 768 pixels) taken from 32 benign 
% and 26 malignant breast cancer patients. The learning task is to classify 
% images as benign (negative) or malignant (positive). Patches of 7x7 size 
% are extracted. The image is thresholded to segment the content from the 
% white background and the patches that contain background more than 75% of 
% their area are discarded. The features used are 657 features that are 
% global to the patch (histogram, LBP, SIFT), and averaged features 
% extracted from the cells, detected in each patch."

rng('default') % For reproducibility
warning on all % This will tell you if MaxIter reaches its limit

load(datafile)
data = x.data; % 2002x708 (instances x feature values)

% Exclude features with zero variance
vr = var(data);
data(:,vr < eps) = [];
% Exclude features are linearly correlated to another feature
rho = corr(data);
[row,col] = find(rho == 1);
idx = find(row-col ~= 0,1);
while ~isempty(idx)
  data(:,idx) = [];
  rho = corr(data);
  [row,col] = find(rho == 1);
  idx = find(row-col ~= 0,1);
end

%% Transform the data using PCA

data = data./var(data); % Normalisation, mean centering is included in the pca function
[~,score,latent,~,explained] = pca(data);

figure(2), subplot(2,1,1)
plot(1:100,latent(1:100)) % Kandemir uses the 100 first components
xlabel(sum(explained(1:100)))
title('Scree plots')
% From the scree plot, it is apparent that many of those don't contribute
% We therefore have closer look at the first 10
subplot(2,1,2)
plot(1:10,latent(1:10)) 
xlabel(sum(explained(1:10))) 
drawnow

data = score(:,1); % We actually use just the first component

% The bags are identified from the bag id
[bag_class, x_bags] = bag_identification(x, data);
n_bag = length(x_bags);
neg_idx = find(bag_class == 0);
n_neg = length(neg_idx);
pos_idx = find(bag_class == 1);
n_pos = length(pos_idx);

%%     
% Parameters for the EM-algorithm:  
maxiter = 1000;
reps = 10;
reg = 1e-6; % Avoiding non-invertible matrices
EMparam = [maxiter reps reg];
K = 10; % Maximum number of components 

% Fit a Gaussian mixture model to each of the bags
bags_distr = cell(n_bag,K);
for j = 1: n_bag
  obj = gaussian_mixture(K,x_bags{j},EMparam);
  bags_distr(j,:) = obj;
end

F = 4; % Number of folds, 4 as in the paper of Kandemir. 
T = 10; % Number of repetitions for CV randomisation
AUC = zeros(T,3);
for t = 1: T % the randomization might have an influence     
  % We use F-fold, stratified cross-validation. 
  rng(t)
  n_ind = crossvalind('Kfold',n_neg,F);
  p_ind = crossvalind('Kfold',n_pos,F);
  
  fold = cell(1,F);  
  for f = 1: F
    fold{f} = [neg_idx(n_ind==f) pos_idx(p_ind==f)];
  end
  
  bag2class_div = zeros(3,n_bag);
  for f = 1: F % The folds of cross-validation
        
    x_neg = []; % The negative class
    x_pos = []; % The positive class
      
    % The negative bags in the training set
    for j = setdiff(neg_idx,fold{f}) 
      x_neg = [x_neg; x_bags{j}];
    end      
    
    % The positive bags in the training set
    for j = setdiff(pos_idx,fold{f}) 
      x_pos = [x_pos; x_bags{j}];
    end        
      
    %% Fit the distribution to the classes
     
    % Fit a Gaussian mixture model to all negative instances
    neg_distr = gaussian_mixture(K,x_neg,EMparam);
    AIC = zeros(1,K);
    for k = 1: K
      AIC(k) = neg_distr{k}.AIC;
    end
    [~,k_AICneg] = min(AIC);
    neg_distr = neg_distr{k_AICneg};
      
    % Fit a Gaussian mixture model to all positive instances
    pos_distr = gaussian_mixture(K,x_pos,EMparam);
    AIC = zeros(1,K);
    for k = 1: K
      AIC(k) = pos_distr{k}.AIC;
    end 
    [~,k_AICpos] = min(AIC);
    pos_distr = pos_distr{k_AICpos};
    
    % Fit the bag GMMs. Find minimum AIC. 
    AIC = zeros(1,K); k_AIC = zeros(1,K);
    distr = bags_distr(fold{f},:); % The bags of the test set
    nf_bag = size(bags_distr(fold{f},:),1); % number of bags in the test set
    obj = cell(1, nf_bag);

    for i = 1: nf_bag
      for k = 1: K
        AIC(k) = distr{i,k}.AIC;
      end
      [~,k_AIC(i)] = min(AIC);
      obj{i} = distr{i,k_AIC(i)};
    end
    bags_f = obj;
    
    bag2class_div(:,fold{f}) = bag_to_class_divergence(neg_distr,pos_distr,bags_f);
  end
    
  rBH = bag2class_div(1,:);
  rKL = bag2class_div(2,:);
  cKL  = bag2class_div(3,:);
    
  % Classification. Simple threshold.
    
  AUC(t,1) = AUC_ROC(rBH,bag_class);
  AUC(t,2) = AUC_ROC(rKL,bag_class);
  AUC(t,3) = AUC_ROC(cKL,bag_class);
  AUC(t,:)
end

mean(AUC)
 



