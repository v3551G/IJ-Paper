classdef RedwineDataModel < DataModel
    
    methods
        function obj = RedwineDataModel(n, eps)
            obj@DataModel(n,2,eps);
        end
    end
    
    methods (Access = protected)
        function [x, y, z] = generateDataModel(~, n, ~, eps)                        
            xx = csvread('datasets\winequality-red.csv');    
            x = xx(randperm(size(xx, 1), n), :);
            mask = x(:, 12)>6;
            y = nan(size(mask));
            y(mask) = +1;
            y(~mask) = -1;
            x = x(:, 1:11);
            z=y;
        end
    end
end

