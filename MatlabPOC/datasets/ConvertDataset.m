clear all;
close all;
clc

%% Convert table to cell

X = table2cell(glass);


%% All doubles

for i = 1:size(X,1)
    for j = 1:size(X,2)
        X{i,j} = double(X{i,j});
    end
end

%% Cell 2 mat

X = cell2mat(X);

%% Convert Y values

X(X(:,end) == 9,end) = 5;
%X(X(:,end) == 2,end) = 1;

%% X and Y + save

Y = X(:,end);
X = X(:,1:end-1);


save('glass.mat','X','Y')
