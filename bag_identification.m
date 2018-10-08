function [bag_class, x_dbag, size_bag] = bag_identification(x, data)

% if bag ids are given as character vector
if ischar(x.ident.milbag)
  bag_id_cell = cellstr(x.ident.milbag);
  bag_name = unique(bag_id_cell);
  bag_id = zeros(1, length(bag_name));
  % convert to numbers
  for i = 1: length(bag_name)
    bag_id(ismember(bag_id_cell, bag_name{i})) = i;
  end
else
  bag_id = x.ident.milbag;
end
    
n_bag = length(unique(bag_id)); 
                                                           
% The bags are identified from the bag id and given its class label
class = x.nlab; % each instance has the class label of the bag
bag_class = zeros(1,n_bag);
size_bag = zeros(1,n_bag);
start = zeros(1,n_bag);
stop = zeros(1,n_bag);
for j = 1: n_bag
  start(j) = find(bag_id == j,1);
  stop(j) = find(bag_id == j,1,'last');
  size_bag(j) = stop(j)-start(j)+1;
  bag_class(j) = class(start(j));
end
bag_class = bag_class-1; % 0-1 class labels

% These are my bags. 
x_dbag = cell(1,n_bag);
for j = 1: n_bag
  bag = data(start(j):stop(j),:); 
  x_dbag{j} = bag; 
end

