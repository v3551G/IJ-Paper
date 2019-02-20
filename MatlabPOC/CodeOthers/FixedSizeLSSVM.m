function [accTrain,accTest,time] = FixedSizeLSSVM(X,Y,Xt,Yt,kernel,h)


tic

type = 'classification';
crit_old=-inf;
Nc=h;
Xs=X(1:Nc,:);
Ys=Y(1:Nc,:);
sigma2ent = 0.1;

%
% iterate over data
%
for tel=1:5*length(X)
   
  
  %
  % new candidate set
  %
  Xsp=Xs; Ysp=Ys;
  S=ceil(length(X)*rand(1));
  Sc=ceil(Nc*rand(1));
  Xs(Sc,:) = X(S,:);
  Ys(Sc,:) = Y(S);
  Ncc=Nc;

  %
  % automaticly extract features and compute entropy
  %
  crit = kentropy(Xs,kernel, sigma2ent);
  
  if crit <= crit_old
    crit = crit_old;
    Xs=Xsp;
    Ys=Ysp;
  else
    crit_old = crit;
  end
    
end
time = toc;

% Train LS-SVM classifier
L_fold = 10; % L-fold crossvalidation
[gam,sig2] = tunelssvm({Xs,Ys,type,[],[],kernel},'simplex','crossvalidatelssvm',{L_fold,'misclass'});
[alpha,b] = trainlssvm({Xs,Ys,type,gam,sig2,kernel});

% Performace
Ytrain = simlssvm({Xs,Xs,type,gam,sig2,kernel,'preprocess'},{alpha,b},X);
Ytest = simlssvm({Xs,Xs,type,gam,sig2,kernel,'preprocess'},{alpha,b},Xt);

accTrain = sum(Ytrain == Y)/length(Y);
accTest = sum(Ytest== Yt)/length(Yt);


end

