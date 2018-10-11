function data = data_cleansing(loadfile)

load(loadfile)

data = x.data; 

warning on all % This will tell you if MaxIter reaches its limit
                
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

[bag_class, x_dbag, bag_size] = bag_identification(x, data);

% Reduce number of instances if it is 3 times the median
msiz = median(bag_size);
idx = find(bag_size> 3*msiz);
for i = idx
  y = randsample(bag_size(i), 3*msiz);
  x_dbag{i} = x_dbag{i}(y,:);
  bag_size(i) = 3*msiz;
end

