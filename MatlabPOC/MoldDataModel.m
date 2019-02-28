classdef MoldDataModel < DataModel
    
    methods
        function obj = MoldDataModel(n, eps)
            obj@DataModel(n,2,eps);
        end
    end
    
    methods (Access = protected)
        function [x, y, z] = generateDataModel(~, n, ~, eps)                        
            xx = csvread('datasets\gp.csv');    
            yy = csvread('datasets\bp.csv');    
            x=[xx; yy];
            y=[+1*ones(size(xx, 1), 1); -1*ones(size(xx, 1), 1)];
            z=y;
        end
    end
end

