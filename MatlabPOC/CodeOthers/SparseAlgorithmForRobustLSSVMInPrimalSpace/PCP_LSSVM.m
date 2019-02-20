function [BS,P]=PCP_LSSVM(train, numb_tr, kerhp, subsetsize,errorbound)
%%Solve the P-LSSVM by pivoted Cholesky decomposing kernel matrix, 
%where the basic set is chosen by pivoting the maximum diag of Schur complement. 
%The whole kernel matrix is approximated by P*P'. 
% This version is better than the algorithm in"On the low-rank
% approximation by the pivoted Cholesky decomposition".
%%
%Input:
%  train----------training samples matrix, (one row one sample)
%  numb_tr----------the number of training data
%  kerhp--------kernel parameters.
%  subsetsize-----random subset size.
%  errorbound-----approximation error bound on trace norm
%Output:

%numb_tr=size(train,1);   
r=1;
BS=zeros(0,subsetsize);
N= 1:numb_tr;
d_K=ones(numb_tr,1);      %The diag the kernel matrix of gaussian, for other kernel 
                              %  it should be changed
%I=randperm(m);d_K(I(1))=d_K(I(1))+1;%%make the first random selection.
error(r) = sum(d_K);
P=zeros(numb_tr,0);
while  error(r)>errorbound &&r<=subsetsize%
    %find the maximum diag of Shur compliment:
    [~,index]=max(d_K(N));s_in=N(index);N(index)=[];      
    k_in=Kgaussian(kerhp,train,train(s_in,:));
    if r==1
        p=k_in/sqrt(k_in(s_in)); 
    else
        u=P(s_in,:)';nu=sqrt(k_in(s_in)-u'*u);
        p=(k_in-P*u)/nu;
        p(BS)=0;
    end
    P(:,r)=p; BS(r)=s_in;
    d_K(N)=d_K(N)-p(N).^2;
    error(r+1) = sum(d_K(N));
    r=r+1;
end
return
 


