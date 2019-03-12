
classdef YinYangDataModel < DataModel
    %Tau's data model, ref [5] 
    %At last a toy problem in reference [5], which is a binary
%     classification problem, was employed to show the difference of SV
%     distribution between ssLS-SVM, psLS-SVM and csLS-SVM. The 250
%     data points of the first class are randomly generated from normal
%     distribution with central vector [?0.3, ?0.3] and covariance matrix
%     0.25I, the 250 training data points of the second class are randomly
%     generated from normal distribution with central vector [0.3, 0.3] and
%     covariance matrix 0.25I, where I is a 2×2 identity matrix.

    properties (Access = private)
        delta = 2.5;
    end
    
    methods
        function obj = YinYangDataModel(n, eps)
            obj@DataModel(n, 2,eps);
        end
    end
    
    methods (Access = public)
        
        function fn = getfilename(~)
           fn = 'yydatamodel';
        end
        
        function [x, y, z] = generateDataModel(this, n, ~, eps)
            nn = floor(n/2);
            m = ceil(eps * n);            
            
            leng = 1; sig = .20;            
            yin = nan(n, 3);
            yang = nan(n, 3);            
            samplesyin = nan(n, 2);
            samplesyang = nan(n, 2);
            for t=1:n
              yin(t,:) = [2.*sin(t/n*pi*leng), 2.*cos(.61*t/n*pi*leng), (t/n*sig)]; 
              yang(t,:) = [-2.*sin(t/n*pi*leng), .45-2.*cos(.61*t/n*pi*leng), (t/n*sig)]; 
              samplesyin(t,:)  = [yin(t,1)+yin(t,3).*randn   yin(t,2)+yin(t,3).*randn];
              samplesyang(t,:) = [yang(t,1)+yang(t,3).*randn   yang(t,2)+yang(t,3).*randn];
            end

            x=[samplesyin(randperm(n, nn), :); samplesyang(randperm(n, nn), :)];
            y=[+1 * ones(nn,1); -1 * ones(nn,1)];
            z=y;            
            oIndices = randperm(n, m);
            
            %%% Outliers: Flip labels;
            y(oIndices)= -1 .* y(oIndices);
            z(oIndices)=0;
        end
    end
end




