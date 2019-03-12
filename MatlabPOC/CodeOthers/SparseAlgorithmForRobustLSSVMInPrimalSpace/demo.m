clear
load usps_12_noise 
%USPS is a muti-class data set with 10 classes. Here a binary classication problem is trained
%to separate class 1 from 2. The numbers of training and test samples are 2199 and 623 respectively, 
%and the dimension is 256.In order to test the robust of our algorithm
%to outliers,we chose 30% of samples that were far away from the decision hyperplane, 
%then randomly sampled 1/3 of them and flipped their labels to simulate outliers.
train_data=full(train_data);
test_data=full(test_data);
train_num=length(addnoise_train_label);test_num=length(test_label);
ker = 2^-7;subsetsize=floor(train_num*0.05);errorbound= 2^(-3);lam =10^0;tau=1.1; rou=10^(-2);itermax=100;    
[acc,trtime,tetime,nsv]=SR_LSSVM(train_data,addnoise_train_label,train_num,test_data,test_label,test_num,ker, subsetsize,errorbound,lam,tau,rou,itermax);
