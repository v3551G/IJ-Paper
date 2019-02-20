function [perform,trtime,tetime,nsv]=SR_LSSVM(train,train_label,n,test,test_label,test_num,kerhp, subsetsize,errorbound,lam,tau,rou,itermax)
%%Input
  %train: training sample
  %train_label:training label
  %n:the number of training data
  %test: test sample
  %test_label: test label
  %test_num: the number of test data
  %kerhp:the parameter in the Gaussian kernel function
  %subsetsize: random subset size.
  %errorbound:approximation error bound on trace norm
  %lam:regularization parameter
  %tau:the truncated parameter
  %rou£ºthe stop criterion, a very small positive number
  %itermax:the maximum number of iterations
%%Output
  %perform:the accuracy for classification dataset
  %trtime:training time
  %tetime: test time
  %nsv: the number of support vectors
tic;
[BS,P]=PCP_LSSVM(train, n,kerhp, subsetsize,errorbound);
m=size(BS,2);
s=zeros(n,1);olds=s+0.02;
t=1;
p=10^4;
tmp=sum(P);
L=chol(lam*eye(m)+P'*P-tmp'*(tmp/n));
while norm(s-olds)>rou && t<=itermax
    y=train_label-s;    
    tmp1 =L\(L'\(P'*(y-sum(y)/n)));
    a = P(BS,:)'\tmp1;
    b = (sum(y)-tmp*tmp1)/n;
    olds=s;
    r1=train_label-P*(P(BS,:)'*a)-b;
    s=r1.*min(1,exp(p*(r1.^2-tau.^2)))./(1+exp(-p*abs(r1.^2-tau.^2)));
    t=t+1;    
end
trtime=toc;
%% find support vectors and count the number of them
I=logical(abs(a)>10^-4);
nsv=nnz(I);
%% test   
tic;
f=sign(Kgaussian(kerhp,train(BS,:),test(:,:))'*a+b);
perform=sum(test_label==f)/test_num;
tetime=toc;
