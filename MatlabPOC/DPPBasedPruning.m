classdef DPPBasedPruning < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
        function idx = selectSVs(this, kModel)
            % build L-ensemble matrix (typically L = Gaussian RBF)
            L = kModel.compute(X,X);
            C = decompose_kernel(L);
            idx = sample_dpp(C,h);
        end  
    end
    
end

