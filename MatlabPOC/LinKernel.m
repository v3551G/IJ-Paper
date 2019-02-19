classdef LinKernel < handle
    
    methods (Access = public)        
        function K = compute(~, Xtrain, Xtest)
            K = Xtrain * Xtest';
        end        
    end
    
end

