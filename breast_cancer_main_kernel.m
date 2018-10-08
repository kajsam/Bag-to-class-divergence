function breast_cancer_main_kernel(datafile)

% Requires: bag_identification.m, AUC_ROC.m


% Data from http://www.miproblems.org/datasets/ucsb-breast/

% "UCSB Breast is an image classification problem. The original datasets 
% consists of 58 TMA image excerpts (896 x 768 pixels) taken from 32 benign 
% and 26 malignant breast cancer patients. The learning task is to classify 
% images as benign (negative) or malignant (positive). Patches of 7x7 size 
% are extracted. The image is thresholded to segment the content from the 
% white background and the patches that contain background more than 75% of 
% their area are discarded. The features used are 657 features that are 
% global to the patch (histogram, LBP, SIFT), and averaged features 
% extracted from the cells, detected in each patch."

% I have tried some different approaches in 'breast_cancer_fold.m', and
% this is what I have settled at

% Load the data

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
neg_idx = find(bag_class == 0);
n_neg = length(neg_idx);
pos_idx = find(bag_class == 1);
n_pos = length(pos_idx);

%%    
% These bandwidhts are taken from the simulation study, adjusted for the
% bag size (50 in the simulation study)
kernel= 'normal';  %'epanechnikov'; % 
bandwidth = 0.5; 

F = 4; % 4 as in the paper. 
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

  cKL = zeros(1,n_neg+n_pos);
  KL_neg = zeros(1,n_neg+n_pos);
  KL_pos = zeros(1,n_neg+n_pos);
  BH_neg = zeros(1,n_neg+n_pos);
  BH_pos = zeros(1,n_neg+n_pos);
  
  for f = 1: F % The folds of cross-validation
    
    x_neg = []; % The negative class
    x_pos = []; % The positive class
                 
    for j = setdiff(neg_idx,fold{f})
      x_neg = [x_neg; x_bags{j}];
    end      

    for j = setdiff(pos_idx,fold{f})
      x_pos = [x_pos; x_bags{j}];
    end        
    
    for j = 1: length(fold{f})
      z = [x_neg; x_pos; x_bags{fold{f}(j)}];
      z = sort(z(:));
      
      % The importance distribution is calculated. 
      f_imp = ksdensity(z,z,'Kernel', kernel); 
      f_imp(f_imp<eps) = eps; % To avoid Inf
      n = length(f_imp);
          
      %% Start kernel density estimation

      % Kde for all negative instances  
      [f_neg,~] = ksdensity(x_neg(:),z,'Kernel', kernel); 
      f_neg(f_neg<eps) = eps;
      
      % Kde for all positive instances
      [f_pos,~] = ksdensity(x_pos(:),z,'Kernel', kernel); 
      f_pos(f_pos<eps) = eps;

      % Kde for the bag
      [f_bag,~] = ksdensity(x_bags{fold{f}(j)},z,'Kernel', kernel); 
      f_bag(f_bag<eps) = eps;
       
      % Calculate each term
      bh_terms_neg = sqrt(f_bag.*f_neg)./f_imp;
      bh_terms_pos = sqrt(f_bag.*f_pos)./f_imp;
      
      kl_terms_neg = f_bag.*log(f_bag./f_neg)./f_imp;
      kl_terms_pos = f_bag.*log(f_bag./f_pos)./f_imp;
      kl_terms_neg(f_bag < eps) = eps; % 0log0 = 0
      kl_terms_pos(f_bag < eps) = eps; 
      
      w = f_pos./f_neg; % 
      ckl_terms = w.*kl_terms_neg;
      ckl_terms(f_pos < eps) = eps; % 0log0 = 0
  
      cKL(fold{f}(j))    = sum(ckl_terms)/n;
      KL_neg(fold{f}(j)) = sum(kl_terms_neg)/n;    
      KL_pos(fold{f}(j)) = sum(kl_terms_pos)/n;
      BH_neg(fold{f}(j)) = -log(sum(bh_terms_neg)/n);
      BH_pos(fold{f}(j)) = -log(sum(bh_terms_pos)/n);
    end  
  end
  rKL = KL_neg./KL_pos;
  rBH = BH_neg./BH_pos;
  
  % Classification. Simple threshold.
  AUC(t,1) = AUC_ROC(rBH, bag_class);
  AUC(t,2) = AUC_ROC(rKL, bag_class);
  AUC(t,3) = AUC_ROC(cKL, bag_class);
end
 
[mean(AUC); sqrt(var(AUC))]

