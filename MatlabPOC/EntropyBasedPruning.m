classdef EntropyBasedPruning < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        nbrOfSupportVectors;
        kModel;
        roi = 0.5;
    end
    
    methods
        
        function this = EntropyBasedPruning(nbrOfSupportVectors, kModel)
            %%%%    nbrOfSupportVectors is split in two, so it must be
            %%%%    dividable by 2!
            assert(mod(nbrOfSupportVectors, 2)==0);
            this.nbrOfSupportVectors = nbrOfSupportVectors / 2;
            this.kModel = kModel;
        end
        
        function svsSolutionsIndices = prune(this, x, a)            
            [~, candidateIndices] = sort(a, 'descend');
            candidateIndices(ceil(numel(candidateIndices)*this.roi):end)=[];
            
            svs = x(candidateIndices(1:this.nbrOfSupportVectors), :);
            svsSolutionsIndices = candidateIndices(1:this.nbrOfSupportVectors);
            
            prevCrit = -inf;                        
            for iteration = 1:10*size(x,1)
               svsCopy = svs;   
               svsSolutionsIndicesCopy = svsSolutionsIndices;
               
               %%%  Replace one svs in the list
               source = ceil(rand(1) * size(x, 1));
               target = ceil(rand(1) * size(svs, 1));               
               svs(target, :) = x(source, :);               
               svsSolutionsIndices(target) = source;
               
               %%%  Grab kernel matrix of the sv-subset and compute the entropy
               [U, lam] = eig(this.kModel.compute(svs));
               if size(lam,1)==size(lam,2);
                   lam = diag(lam);
               end
               crit = -log((sum(U,1)/ size(U, 1)).^2 * lam);   
               if (crit<=prevCrit)
                   %%%  Entropy is worse, reverse the set
                   svs = svsCopy;
                   svsSolutionsIndices = svsSolutionsIndicesCopy;
               else
                   %%%  Entropy is bettter, adjust criteria;
                   prevCrit = crit;
               end
            end
%             L = k.compute(X);
%             dpp_sample = sample_dpp(decompose_kernel(L), Nc);
        end
    end
    
end

