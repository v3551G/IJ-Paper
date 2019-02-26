classdef CrossDataModel < DataModel
    %NormalDataModel Normal distribution with normal distributed outliers
    %   TODO: adjust conform official shift, point, cluster contamination model
    
    methods
        function obj = CrossDataModel(n, eps)
            obj@DataModel(n,2,eps);
        end
    end
    
    methods (Access = protected)
        function [x, y, z] = generateDataModel(~, N, ~, eps)  
            
            % Parameters
            scale = 10;
            cornerwidth = 2;
            gapwidth = 0.8;
            perCorner = round(N/4);
            nn = floor(N/2);
            m = ceil(eps * N);   
            
            xplusmin = [ones(perCorner,1); -1*ones(perCorner,1); ones(perCorner,1); -1*ones(perCorner,1)];
            yplusmin = [ones(perCorner,1); -1*ones(2*perCorner,1); ones(perCorner,1)];
            horizontal = [xplusmin(1:2:end) * gapwidth + xplusmin(1:2:end) * scale .* rand(N/2,1), ...
                          yplusmin(1:2:end) * gapwidth + cornerwidth * yplusmin(1:2:end) .* rand(N/2,1), ...
                          floor((0:N/2-1)'/(perCorner*.5))];
            vertical = [xplusmin(2:2:end) * gapwidth + cornerwidth * xplusmin(2:2:end) .* rand(N/2,1), ...
                        yplusmin(2:2:end) * gapwidth + yplusmin(2:2:end) * scale .* rand(N/2,1), ...
                        floor((0:N/2-1)'/(perCorner*.5))];            
            x= [horizontal; vertical];
            y= x(:,3);
            y(y== 0) = -1;
            y(y== 2) = -1;
            y(y== 3) = 1;
            x = x(:,1:2);
            z=y;            
            
            %%% Outliers
            oIndices = (N-m:N);
            y(oIndices)= -1 .* y(oIndices);
            z(oIndices)=0;
        end
    end
end