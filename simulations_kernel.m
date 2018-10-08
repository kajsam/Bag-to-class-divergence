function mAUC = simulations_kernel(simul)

% Requires: simulation_param.m, sampling.m, lognormal_sampling.m, 
%           AUC_ROC.m

% Input - simul :   number 1 -6 :   Which of the six different bag 
%                                   distributions to test.
% Output - mAUC:    Mean area under receiver operating curve.


% This function samples bags and corresponding instances for
% the training set, and bags and corresponding instances for the test set.
% Each bag and class is estimated by a kernel density estimation with
% Gaussian kernel and fixed bandwith. Different divergences are then
% calculated based on importance sampling. Finally, classification is done,
% and AUC or SE/SP pairs are reported. 

rng('default')              % for reproducibility  

T = 50;                     % number of repetitions
n_x = 50;                   % number of instances in each bag
n_neg_vec = [5 10 25];      % number of negative bags
n_pos_vec = [1 5 10];       % number of positive bags
n_test = 100;               % number of test bags

imp_bw = 0.5;   % Fixed bandwidths for the kernel estimation
bag_bw = 1;

fig = 0; 

% Different bag distributions. Extracting the fixed parameters. 
[param_n, param_p, flip, P] = simulation_param(simul);

mAUC = zeros(3,length(n_pos_vec),length(n_neg_vec));

for neg_bags =  1: 3
  figure(1), hold on, grid on
  
  for pos_bags = 1: 3 
    
    n_neg = n_neg_vec(neg_bags);     % nr of negative bags of size 1xn_x 
    
    n_pos = n_pos_vec(pos_bags);     % nr of positive bags
    
    N = [n_x n_neg n_pos n_test];    % nr of instances and bags

    pos_bw = imp_bw;
    neg_bw = imp_bw;
    
    if n_pos == 1                   % if only one pos bag, 
      pos_bw = bag_bw;              % then bandwidth as bag bandwidth
    end
        
%    BH_KL = zeros(2,2,T);               % Minimum bag-to-bag sensitivity/specificity pair
    
    % Bag-to-class specificities
    BH_SP = ones(T,ceil(n_test/2)+1);    % Bhattacharyya
    KL_SP = ones(T,ceil(n_test/2)+1);    % Kullback-Leibler
    CC_SP = ones(T,ceil(n_test/2)+1);    % Class-conditional
    
    AUC = zeros(T,3);                   % Area under the curve
    
    for t = 1: T                        % T repetitions bc random sampling
  
      if simul < 5
        X = sampling(N, param_n, param_p, P, flip); % Sampling all the bags and instances
      elseif simul <= 6
        X = lognormal_sampling(N, param_n, param_p, P(1));
      end
      
      x_neg = X{1};         % negative bags
      x_pos = X{2};         % positive bags
      x_test = X{3};        % test bags
      n_test_neg = X{4};    % number of negative test bags
      
      % Sensitivity and specificity calculated according to minimum
      % bag-to-bag divergence classification for BH distance and KL
      % information
      % BH_KL(:,:,t) = simulations_kernel_bag_to_bag(X, imp_bw, bag_bw);
             
      %% Estimation of distributions

      % Each distribution is approximated by a kernel density estimation
      % with Epanechnikov kernel and fixed bandwidth.

      n_test = size(x_test,1);

      %% Approximations of integrals by importance sampling

      % Calculate the BH distance,the KL information and the class
      % conditional dissimilarity

      KL_cc = zeros(1,n_test);  % Class conditional KL information
      BH_neg = zeros(1,n_test); % Bhattacharyya bag-neg
      BH_pos = zeros(1,n_test); % Bhattacharyya bag-pos
      KL_neg = zeros(1,n_test); % KL bag-to-negative
      KL_pos = zeros(1,n_test); % KL bag-to-positive
      
      if fig 
        clf(figure(4)),hold on, title('Fitted test bags')  % Shows fitted bags
        x = (-20:0.1:20)';
      end
      
      for j = 1: n_test  
  
        % The importance sample consists of all instances in the training set and
        % the instances in the bag up for classification. 
        z = [x_neg; x_pos; x_test(j,:)]; 
        z = sort(z(:));
        
        % The importance distribution is estimated. 
        f_imp = ksdensity(z,z, 'Kernel', 'epanechnikov','Bandwidth',imp_bw);
        f_imp(f_imp<=eps) = eps; % To avoid Inf
        n = length(f_imp);
    
        %% Start kernel density estimation

        % Kde for all negative instances  
        f_neg = ksdensity(x_neg(:),z, 'Kernel', 'epanechnikov','Bandwidth',neg_bw);
        f_neg(f_neg<=eps) = eps;
        
        % Kde for all positive instances
        f_pos = ksdensity(x_pos(:),z, 'Kernel', 'epanechnikov','Bandwidth',pos_bw);
        f_pos(f_pos<=eps) = eps;
        
        % Kde for unlabelled bag
        f_bag = ksdensity(x_test(j,:),z, 'Kernel', 'epanechnikov','Bandwidth',bag_bw);
        f_bag(f_bag<=eps) = eps;
        
        if fig 
          if j < 11
            [y,~] = ksdensity(x_test(j,:),x);
            plot(x,y,'b')
          end
          if (j > n_test_neg) && (j < n_test_neg+11)
            [y,~] = ksdensity(x_test(j,:),x);
            plot(x,y,'r')
          end
        end
  
        % Calculate each term for the imp samp sum.
        bh_terms_neg = sqrt(f_bag.*f_neg)./f_imp; 
        bh_terms_pos = sqrt(f_bag.*f_pos)./f_imp;
        kl_terms_neg = f_bag.*log(f_bag./f_neg)./f_imp;
        kl_terms_pos = f_bag.*log(f_bag./f_pos)./f_imp;
        w = f_pos./f_neg;
        cc_terms = w.*kl_terms_neg;
 
        BH_neg(j) = -log(sum(bh_terms_neg)/n);
        BH_pos(j) = -log(sum(bh_terms_pos)/n);
        KL_neg(j) = sum(kl_terms_neg)/n;    
        KL_pos(j) = sum(kl_terms_pos)/n;
        KL_cc(j) = sum(cc_terms)/n;
      end
      
      BH_neg(BH_neg < eps) = eps;
      BH_pos(BH_pos < eps) = eps;
      BH = BH_neg./BH_pos; % Classification by ratio
      
      KL_neg(KL_neg < eps) = eps;
      KL_pos(KL_pos < eps) = eps;
      KL = KL_neg./KL_pos;
      
      if fig 
        clf(figure(3)), hold on, title('Fitted distributions') % Shows fitted distr

        [y,~] = ksdensity(x_neg(:),x);
        plot(x,y,'b')
        [y,~] = ksdensity(x_pos(:),x);
        plot(x,y,'r')
        legend('Negative class','Positive class','Location','NE')
      end

      class_vec = zeros(1,n_test);
      class_vec(n_test_neg+1:end) = 1;

      % Classification
      [AUC_BH, SP] = AUC_ROC(BH,class_vec);
      BH_SP(t,:) = SP;
      [AUC_KL, SP] = AUC_ROC(KL,class_vec);
      KL_SP(t,:) = SP;
      [AUC_CKL, SP] = AUC_ROC(KL_cc,class_vec);
      CC_SP(t,:) = SP;
      AUC(t,:) = [AUC_BH AUC_KL AUC_CKL];
    end
    
    % The sensitivity is common for all repetitions/classifiers, since this
    % is how the specificity threshold is set
    SE = 0:1/ceil(n_test/2):1; 
    
    % Mean specificity over all repetitions
    mBH_SP  = mean(BH_SP,1);
    mKL_SP  = mean(KL_SP,1);
    mCC_SP = mean(CC_SP,1);
     
    % Mean sensitivity/specificity pair for bag-to-bag classifier
%     bag_SEP = mean(BH_KL,3);
    mAUC(pos_bags,:,neg_bags) = mean(AUC);
    
    dark_green = [0.1 0.5 0.1];
    if pos_bags == 1 || pos_bags >= 4
      p1 = plot(1-mBH_SP,SE,'.-r'); 
      p2 = plot(1-mKL_SP,SE,'.-','Color', dark_green);
      p3 = plot(1-mCC_SP,SE,'.-b');
%       p4 = plot(1-bag_SEP(1,2),bag_SEP(1,1),'*r');
%       p5 = plot(1-bag_SEP(2,2),bag_SEP(2,1),'*g');
    elseif pos_bags == 2
      p6 = plot(1-mBH_SP,SE,'--r'); 
      p7 = plot(1-mKL_SP,SE,'--g');
      p8 = plot(1-mCC_SP,SE,'--b');
%       p9 = plot(1-bag_SEP(1,2),bag_SEP(1,1),'or');
%       p10 = plot(1-bag_SEP(2,2),bag_SEP(2,1),'og');
    elseif pos_bags == 3
      p11 = plot(1-mBH_SP,SE,'r'); 
      p12 = plot(1-mKL_SP,SE,'Color', dark_green);
      p13 = plot(1-mCC_SP,SE,'b');
%       p14 = plot(1-bag_SEP(1,2),bag_SEP(1,1),'xr','MarkerSize',15);
%       p15 = plot(1-bag_SEP(2,2),bag_SEP(2,1),'x','Color',dark_green,'MarkerSize',15);
      p16 = plot([0 0], [0.01 0.01],'--k');
      p17 = plot([0 0], [0.01 0.01],'k');
      xlabel(T)
    end
    
    title(num2str([simul n_neg n_pos]))
    drawnow
  end
%  legend([p13 p12 p11 p15 p14 p16 p17],'cKL', 'rKL', 'rBH', ...
%     'bag-to-bag KL', 'bag-to-bag BH', 'pos bag = 1', 'pos bag = 10', 'Location', 'southeast')
% legend('boxoff')
drawnow
end

