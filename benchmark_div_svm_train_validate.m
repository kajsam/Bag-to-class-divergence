function benchmark_div_svm_train_validate(loadfile)

% Requires: bag_identification.m, data_cleansing.m, importance_sampling.m, 

T = 10; % Number of repetitions for CV randomisation. 5 in Cheplygina.
F = 10; 
% One part to create p_neg and p_pos. One part for D(p_bag, p_neg) that will
% train the classifier. One fold for pure testing. 

% Load the data
load(loadfile)

% Clean the data 
data = data_cleansing(loadfile);

% Get the data sorted by bag, and attach the correct bag label
[bag_class, x_dbag] = bag_identification(x, data);

n_bag = length(bag_class);
n_neg = sum(bag_class == 0); % Number of negative bags
n_pos = sum(bag_class == 1); % Number of positive bags
neg_idx = find(bag_class == 0);
pos_idx = find(bag_class == 1);

ACC = zeros(T,3);

% The divergences for the test sets are obtained by leave-one-out for the
% entire data set.
[rBH, rKL, cKL] = divergences_one(x_dbag, bag_class);


for t = 1: T % the randomization might have an influence     
  
  rng(t) % For reproducibility
  
  % Statified cross-validation
  n_ind = crossvalind('Kfold',n_neg,F);
  p_ind = crossvalind('Kfold',n_pos,F);
  fold = cell(1,F);  
  for f = 1: F
    fold{f} = [neg_idx(n_ind==f) pos_idx(p_ind==f)];
  end
    
  label_rBH = zeros(1,n_bag);
  label_rKL = zeros(1,n_bag);
  label_cKL = zeros(1,n_bag);
  
  for f = 1: F
       
    test_set = fold{f};
    train_set = setdiff(1:n_bag,test_set);
   
    bags_train = x_dbag(train_set);
    
    %%%%%%%%%%%%%%%%%%% Divergences %%%%%%%%%%%%%%%%%%%%
    
    % The divergences for the training set are obtained by leave-one-out
    % for the training set
   
     [svm_rBH, svm_rKL, svm_cKL] = divergences_one(bags_train, bag_class(train_set));
    
  
    %%%%%%%%%%%%%%%%%%% SVM %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % The svm's are trained with the divergences obtained from training
    % data, and the corresponding class labels
    
    class_train = bag_class(train_set);
    
    SVM_BH = fitcsvm(svm_rBH, class_train,'KernelFunction','linear',...
             'Standardize','on','ClassNames',[0 1]);
    SVM_KL = fitcsvm(svm_rKL, class_train,'KernelFunction','linear',...
             'Standardize','on','ClassNames',[0 1]);
    SVM_CKL = fitcsvm(svm_cKL, class_train,'KernelFunction','linear',...
             'Standardize','on','ClassNames',[0 1]);
    
    % Classification
    
    % The bags in the test set are classified
    label_rBH(test_set) = predict(SVM_BH, rBH(test_set,:));
    label_rKL(test_set) = predict(SVM_KL, rKL(test_set,:));
    label_cKL(test_set) = predict(SVM_CKL, cKL(test_set,:));  
    
  end
       
  CP = classperf(bag_class,label_rBH,'Positive', 1, 'Negative', 0); 
  ACC_rBH = (CP.Sensitivity + CP.Specificity)/2;
  
  CP = classperf(bag_class,label_rKL,'Positive', 1, 'Negative', 0); 
  ACC_rKL = (CP.Sensitivity + CP.Specificity)/2;
  
  CP = classperf(bag_class,label_cKL,'Positive', 1, 'Negative', 0); 
  ACC_cKL = (CP.Sensitivity + CP.Specificity)/2;
  
  ACC(t,:) = [ACC_rBH ACC_rKL ACC_cKL];
  
end

ACC
[mean(ACC);
sqrt(var(ACC))]

