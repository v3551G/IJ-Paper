classdef rLSSVM
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        kModel;
        pModel;
    end
    
    methods (Access = private)
       
        function [Kc] = center(~, omega, Kt)
            nb_data = size(omega,1);
            if nargin<3
                onesM = ones(nb_data, nb_data) * (1./nb_data);
                Kc =  omega - onesM * omega - omega * onesM + onesM * omega * onesM;                                
            else
                [l, m] = size(Kt);
                onesMorig = ones(nb_data, nb_data) * (1./nb_data);
                onesMtest = ones(l, m) * (1/m);
                Kc = Kt - onesMtest * omega - Kt * onesMorig + onesMtest * omega * onesMorig;
            end
        end
        
        %%%%    TODO: fatsoeneren
        function weights = sdoWeights(this, x, refinedSubset)
            
            K = this.kModel.compute(x, x);
            
            hindices = find(refinedSubset>0);            
            r = nan(numel(hindices) * (numel(hindices)-1) / 2, size(K, 2)); 
            index = 1;
            for i=1:numel(hindices)-1
                for j=i+1:numel(hindices)
                    gamma = zeros(size(K, 1), 1);
                    gamma(hindices(i))=+1;
                    gamma(hindices(j))=+1;
                    teller = K * gamma;
                    noemer = sqrt(gamma' * K * gamma);
                    r(index, :) = (teller ./ noemer);
                    index = index+1;
                end
            end            
            [~, mindices] = max(abs(r));            
            signal = nan(numel(mindices), 1);            
            for i=1:numel(mindices)
                signal(i) = r(mindices(i), i);
            end

            loc = mean(signal(refinedSubset));
            scat = std(signal(refinedSubset));            
            sdo = abs( signal - loc ) ./ scat;
            before = sum(sdo);
            hmask = 1 - (sdo ./ max(sdo)); % zeros(size(sdo));
            hmask(refinedSubset) = 1;
            
            weights = hmask;
            figure; scatter(x(:, 1), x(:, 2), [], log(weights), 'filled'); grid on;
            colorbar;
            title('obv univ. SDO');
        end
        
        function initialSubsetMask = spatialMedian(this, x, h)
            K = this.kModel.compute(x, x);
            assert(size(K, 1)==size(K, 2));
            n = size(K, 1);
            gamma = ones(n,1)./n;
            for i = 1:15
                w = ones(n,1)./sqrt(diag(K) - 2*K'*gamma + gamma'*K*gamma);
                gamma = w./sum(w);
            end
            % Calculate distance to spatial median
            dist = diag(K) - 2*sum(K.*repmat(gamma',n,1),2) + gamma'*K*gamma;
            %assert(sum((samples - gamma'*samples).^2,2)<=eps);                        
            [~, hIndices] = sort(dist);            
            initialSubsetMask = false(size(hIndices));
            initialSubsetMask(hIndices(1:ceil(h * n))) = true;
        end 
        
       
       function weights = kernelCSteps(this, x, cstepmask, hInitial, hCstep, z)                       
           hRange = hInitial: 0.05:hCstep;
            %%%% Given the initial subset, run kernel C-steps until
            %%%% convergence            
            for hIndex = 1:numel(hRange)            
                h = ceil(hRange(hIndex) * size(x, 1));                
                for iteration = 1:15                    
                    hmask_old = cstepmask;                    
                    
                    Kx = this.kModel.compute(x(cstepmask, :), x(cstepmask, :));                
                    
                    [alphah, Lh] = svd(this.center(Kx));                
                    Ld = diag(Lh);
                    lmask = logical(Ld > 10^(-4));
                    %lmask = logical(Ld > 1.0); % % 10^(-4));
                    Ld = Ld(lmask);
                    alphah = alphah(:, lmask);        
                    alphah = alphah ./ repmat(sqrt(Ld'), size(alphah, 1), 1);   

                    Kt = this.kModel.compute(x, x(cstepmask, :));                                
                    e = this.center(Kx, Kt)*alphah;
                    
                    smd =  size(Kx, 1) * sum((e.^2)./ repmat(Ld', size(e, 1), 1), 2);
                    [~, indices] = sort(smd);

                    cstepmask = false(size(cstepmask));
                    cstepmask(indices(1:h)) = true;
                    
                    if isequal(cstepmask, hmask_old)
                        display(['Convergence at iteration ' mat2str(iteration)]);
                        break;
                    end
                end
            end
            
            %%%%%%%%%   Convert Hard rejection to outlying weights
            %%%
            %%% Stahel-Donoho with cellwise weights -> Hubert cost
            %%% function, eq. 2, pg 3  
            %%% Note that we're not using Stahel-Donoho weights but rather
            %%% the cost function based on mahalanobis distances
            
            scores = smd;
            sdo = max(abs( (scores - repmat(mean(scores(cstepmask, :)), size(scores, 1), 1)) ./ repmat(std(scores(cstepmask, :)), size(scores, 1), 1)), [], 2);
            
            p = sum(lmask);
            c = max(chi2inv(0.5, p), max(sdo(cstepmask)));
            weights = zeros(size(sdo));
            weights(sdo<=c) = 1;
            weights(sdo>c) = (c ./ sdo(sdo>c)).^2; 
            %figure; plot([z, weights]);
       end
       
       function solution = trainWeightedLSSVM(this, x, y, weights, C)
            n = numel(y);
            K = this.kModel.compute(x, x);
            upper = [0, y' ];            
            lower = [y , (y * y') .* K  + diag(weights ./ C)];            
            right = [0; ones(n, 1)];            
            %%%%   Solution is in the form of [b; alphas]            
            solution = ([ upper; lower ] \ right) .* [1; y];    
       end
    end
    
    methods (Access = public)        
        function this = rLSSVM(kModel, pModel)
            this.kModel = kModel;
            this.pModel = pModel;
        end
        
        function train(this, dModel, C)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%    Process +1 class
            w1 = this.trainSingleClass(dModel.x(dModel.y>0, :), dModel.z(dModel.y>0, :));
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%    Process -1 class
            w2 = this.trainSingleClass(dModel.x(dModel.y<0, :), dModel.z(dModel.y<0, :));
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%    Normalize weights to perform class balancing                    
            %w1 = w1 * (max(w2) / max(w1)); %% 100 vs 10 = 10, w1 * 10                        
            w1 = w1 * (sum(w2) / sum(w1)); %% 100 vs 10 = 10, w1 * 10                        
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%    Put everything together
            weights = nan(size(dModel.y));
            weights(dModel.y>0) = w1;
            weights(dModel.y<0) = w2;
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%     Step 4, Solve LS-SVM's SoE            
            solution = this.trainWeightedLSSVM(dModel.x, dModel.y, weights, C);
            
            figure; scatter(dModel.x(:, 1), dModel.x(:, 2), [], solution(2:end), 'filled'); colorbar; title('LS-SVM solution');
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%     Step 5, determine SVS's per class           
            alphas = solution(2:end);
            
            supportVectorMask1 = this.pModel(alphas(dModel.y>0));
            supportVectorMask2 = this.pModel(alphas(dModel.y<0));
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%     Step 6, retrain or info transfer
            %%%            
            %efficency = prunedLsSvmModel.classify(...);
            
            %%%%    That's all, folks!
        end
        
        function weights  = trainSingleClass(this, x, z)
            
            hInitial = 0.5;
            hCstep = 0.75;
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%     Step 1
            %%%
            initisalSubset  = this.spatialMedian(x, hInitial);
            
%             figure; 
%             plot(x(initisalSubset, 1), x(initisalSubset, 2), '.g'); hold on;
%             plot(x(~initisalSubset, 1), x(~initisalSubset, 2), '.r');
%             title('Before');
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%     Step 2
            %%%
            weights = this.kernelCSteps(x, initisalSubset, hInitial, hCstep, z);
            
            figure; 
            scatter(x(:, 1), x(:, 2), [], weights, 'filled'); grid on; colorbar;           
            title('LS-SVM Weight vector'); 
        end
    end
    
    
    
    
end

