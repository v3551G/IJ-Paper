classdef DPPBasedPruning < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        nbrOfSupportVectors;
        kModel;
        %roi = 0.25;
    end
    
    methods
        
        function this = DPPBasedPruning(nbrOfSupportVectors, kModel)
            %%%%    nbrOfSupportVectors is split in two, so it must be
            %%%%    dividable by 2!
            assert(mod(nbrOfSupportVectors, 2)==0);
            this.nbrOfSupportVectors = nbrOfSupportVectors / 2;
            this.kModel = kModel;
        end
        
        function svsSolutionsIndices = prune(this, x, a)            
            
            medValue = median(abs(a));
            indices = find(abs(a)>medValue);
            
            M = this.kModel.compute(x(indices, :));
            L.M = M;
            [V,D] = eig(M);
            L.V = real(V);
            L.D = real(diag(D));
            svsSolutionsIndices = indices(sample_dpp(L, this.nbrOfSupportVectors));
        end
    end
    
end

