classdef PolyKernel < handle
    
    properties (Access = private)
        degree;
    end
    
    methods (Access = public)
        
        function this = PolyKernel(degree)
            this.degree = degree;
        end
        
        function K = compute(this, Xtrain, Xtest)                 
            K = (Xtrain * Xtest' + 1).^this.degree;
        end
    end
    
end

