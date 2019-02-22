classdef NormalDataModel < DataModel
    %NormalDataModel Normal distribution with normal distributed outliers
    %   TODO: adjust conform official shift, point, cluster contamination model
    
    methods
        function obj = NormalDataModel(n, eps)
            obj@DataModel(n,2,eps);
        end
    end
    
    methods (Access = protected)
        function [x, y, z] = generateDataModel(~, n, ~, eps)                        
            
            m=ceil(n*0.5);            
            cRegular1 = mvnrnd([-1 +1], [2 1.5; 1.5 2], m);
            cRegular2 = mvnrnd([+1 -1], [1 0.5; 0.5 1.5], m);
            cOutliers = mvnrnd([-4 +4], 0.15*eye(2), floor(n*eps));
            
            x=[cRegular1; cRegular2];
            y=[+1 * ones(m, 1); -1 * ones(m, 1)];
            mask = randperm(size(x, 1), size(cOutliers, 1));
            
            x(mask, :) = cOutliers;
            z=y;
            z(mask, :) = 0;
        end
    end
end

