function [accTrain,accTest,time] = WeightedRobustLSSVM(X,Y,Xt,Yt,kernel,h)

tic

X_subset = X;
Y_subset = Y;

h_list = N:floor(N*0.05):h;

for h_current = 2:length(h_list)
    
    % Train robust model
    model = initlssvm(X_subset,Y_subset,'f',[],[],kernel);
    L_fold = 10; %10 fold CV
    model = tunelssvm(model,'simplex','rcrossvalidatelssvm',{L_fold,'mae'},'whuber');
    model = robustlssvm(model);
    
    % Prune
    alpha = model.alpha;
    [~,idx] = sort(abs(alpha),'descend');
    idx = idx(1:h_current);
    X_subset = X_subset(idx,:);
    Y_subset= Y_subset(idx,:);
end

% Train robust model
model = initlssvm(X_subset,Y_subset,'f',[],[],kernel);
L_fold = 10; %10 fold CV
model = tunelssvm(model,'simplex','rcrossvalidatelssvm',{L_fold,'mae'},'whuber');
model = robustlssvm(model);
    
time = toc;

% Performance
% Model parameters
sigma_R = model.kernel_pars;
alpha_R = model.alpha;
b_R = model.b;

% Kernel matrices
K= kernel_matrix(X_subset,kernel,sigma_R,X);
Kt = kernel_matrix(X_subset,kernel,sigma_R,Xt);

% Predict
Ytrain = sign(K'*alpha_R + b_R);
Ytest = sign(Kt'*alpha_R + b_R);

accTrain = sum(Ytrain == Y)/length(Y);
accTest = sum(Ytest== Yt)/length(Yt);

end

