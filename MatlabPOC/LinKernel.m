classdef LinKernel < handle
    
    methods (Access = public)        
        function K = compute(~, Xtrain, Xtest)
            if nargin<3
                Xtest = Xtrain;
            end    
            K = Xtrain * Xtest';
        end        
    end
    
end

