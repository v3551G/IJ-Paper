classdef EntropyBasedPruning < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
        function selectSVs(this, kModel)
           % Start random and optimize
            N = size(X,1);
            idx = randperm(N,h);
            crit_old = kentropy(X(idx,:),kModel);
            for tel=1:5*N
                % New candidate set
                idx_old = idx; 
                S =ceil(N*rand(1));
                Sc =ceil(h*rand(1));
                idx(Sc) = S;

                % Compute entropy
                crit = kentropy(X(idx,:),kModel);
                if crit <= crit_old  
                    idx = idx_old;
                else
                    crit_old = crit;
                end
            end
        end

        function en = kentropy(X, kModel)
            n= size(X,1);
            [U,lam] = svd(kModel.compute(X,X));
            en = -log((sum(U,1)/n).^2 * lam);
        end 
    end
    
end

