

clc;
close all;
clear all;

%rng default;

kModel = RbfKernel(1);        
dModel = YinYangDataModel(1000, 0.1);         
X = dModel.x(dModel.y>0, :);

Nc=20;
gamma = 0.1;

k = RbfKernel(1/10);
%%%%%%%%%%%%%%%%%%%%%%%%%%  deftige implementatie

svs = X(1:Nc, :);
prevCrit = -inf;
entropy = nan(5*size(X, 1), 1);
counter = 1;
for iteration = 1:50*size(X, 1)
   svsCopy = svs;   
   
   %%%  Replace one svs in the list
   svs(ceil(rand(1) * size(svs, 1)), :) = X(ceil(rand(1) * size(X, 1)), :);
   %%%  Grab kernel matrix of the sv-subset and compute the entropy
   [U, lam] = eig(k.compute(svs));
   if size(lam,1)==size(lam,2);
       lam = diag(lam);
   end
   crit = -log((sum(U,1)/ size(U, 1)).^2 * lam);   
   if (crit<=prevCrit)
       %%%  Entropy is worse, reverse the set
       svs = svsCopy;
   else
       %%%  Entropy is bettter, adjust criteria;
       prevCrit = crit;
       entropy(counter) = crit;
       counter = counter +1;
   end
   
end

figure; plot(entropy);
figure; 
plot(X(:, 1), X(:, 2), '.'); hold on;
plot(svs(:, 1), svs(:, 2), '*r'); 
title('ours');







