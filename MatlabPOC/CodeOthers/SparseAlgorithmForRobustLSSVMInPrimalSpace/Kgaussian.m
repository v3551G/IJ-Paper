function K=Kgaussian(kerhp,train1,train2,train1_norm2)
K = train1*train2';
if nargin>3
    normY = sum(train2.^2,2);%train_2的2-范数的平方
    K = exp(-kerhp*(bsxfun(@plus,train1_norm2,normY') - 2*K));
else
    normX = sum(train1.^2,2);
    normY = sum(train2.^2,2);
    K = exp(-kerhp*(bsxfun(@plus,normX,normY') - 2*K));
end


% function K=Kgaussian(kerhp,train1,train2,normX)
% K = train1*train2'; %nargin是用来判断输入变量个数的函数
% if nargin<3,    normX = sum(train1.^2,2);end
% normY = sum(train2.^2,2);
% K = exp(-kerhp*(bsxfun(@plus,normX,normY') - 2*K));