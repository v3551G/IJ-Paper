classdef rLSSVM < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (Access = private)
        kModel;
        pModel;
        supportVectorData;
        prunedAlphas;
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
        
       
       function weights = kernelCSteps(this, x, cstepmask, hInitial, hCstep)                       
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
            weights = false(size(sdo));
            weights(sdo<=c) = true; 
            weights(sdo>c) = false; % (c ./ sdo(sdo>c)).^2;                         
       end
       
       function solution = trainWeightedLSSVM(this, xTrain, yTrain, C)
            %%%%   Solution is in the form of [b; alphas]         
            n = numel(yTrain);
            K = this.kModel.compute(xTrain, xTrain);
            upper = [0, yTrain' ];
            lower = [yTrain , (yTrain * yTrain') .* K  + (1 ./ C) * eye(n,n) ];
            right = [0; ones(n, 1)];                            
            solution = ([ upper; lower ] \ right) .* [1; yTrain];
       end
    end
    
    methods (Access = public)        
        function this = rLSSVM(kModel, pModel)
            this.kModel = kModel;
            this.pModel = pModel;
        end
        
        function train(this, dModel, C)
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%    How many outliers do we expect in out dataset
            hInitial = 0.5;
            hCstep = 0.80;
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%    Process +1 class
            w1 = this.trainSingleClass(dModel.x(dModel.y>0, :), hInitial, hCstep);
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%    Process -1 class
            w2 = this.trainSingleClass(dModel.x(dModel.y<0, :), hInitial, hCstep);
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%    Normalize weights to perform class balancing                    
            %w1 = w1 * (max(w2) / max(w1)); %% 100 vs 10 = 10, w1 * 10                        
            %w1 = w1 * (sum(w2) / sum(w1)); %% 100 vs 10 = 10, w1 * 10                        
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%    Put everything together
            weights = false(size(dModel.y));
            weights(dModel.y>0) = w1;
            weights(dModel.y<0) = w2;            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%     Step 4, Solve LS-SVM's SoE            
            solution = this.trainWeightedLSSVM(dModel.x(weights, :), dModel.y(weights), C);
            alphas = zeros(size(weights));
            alphas(weights) = solution(2:end);
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%     Step 5a, determine SVS's per class            
            figure;             
            scatter(dModel.x(weights, 1), dModel.x(weights, 2), [], alphas(weights), 'filled'); 
            colorbar; title('Hard rejection weighted LS-SVM alphas'); colormap(bluewhitered);
            
            c1 = find(dModel.y>0 & weights & alphas>0);
            c11 = dModel.y>0;
            c2 = find(dModel.y<0 & weights & alphas<0);
            c21 = dModel.y<0;
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%
            %%%     TODO: Step 5b, balance alphas to compensate for class skew            
            
            %%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            svIndices1 = this.pModel.prune(dModel.x(c1, :), abs(alphas(c1)));
            svIndices2 = this.pModel.prune(dModel.x(c2, :), abs(alphas(c2)));
            
            figure; 
            plot(dModel.x(:, 1), dModel.x(:, 2), '.'); hold on;
            plot(dModel.x(c1(svIndices1), 1), dModel.x(c1(svIndices1), 2), '*r');  
            plot(dModel.x(c2(svIndices2), 1), dModel.x(c2(svIndices2), 2), '*g');  grid on;
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%     Step 6, info transfer
            %%%            
            
            K = this.kModel.compute(dModel.x);
            
            mask = false(size(dModel.y));
            mask(c11) = true;
            mask(svIndices1) = false;
            
            delta_alpha = pinv(K(:, c1(svIndices1)))*K(:,mask)*alphas(mask, :);
            svAlphas1 =  alphas(c1(svIndices1), :) + delta_alpha;
            
            mask = false(size(dModel.y));
            mask(c21) = true;
            mask(svIndices2) = false;
            
            delta_alpha = pinv(K(:,c2(svIndices2)))*K(:,mask)*alphas(mask, :);
            svAlphas2 =  alphas(c2(svIndices2), :) + delta_alpha;            
            
            this.supportVectorData = [dModel.x(c1(svIndices1), :); dModel.x(c2(svIndices2), :);];
            this.prunedAlphas = [solution(1); svAlphas1; svAlphas2];
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%    That's all, folks!            
        end
        
        function weights  = trainSingleClass(this, x, hInitial, hCstep)
            initisalSubset  = this.spatialMedian(x, hInitial);
            weights = this.kernelCSteps(x, initisalSubset, hInitial, hCstep);            
        end
    
        function plot(this, dModel)
            [rr, cc] = meshgrid(-4:0.01:4);    
            output = this.predict([rr(:), cc(:)]);    
            z=reshape(output, size(rr)); 
            figure; 
            contourf(rr, cc, z, [0 0]); hold on;
            if nargin==2
                plot(dModel.x(:, 1), dModel.x(:, 2), '.');
            end
            plot(this.supportVectorData(:, 1), this.supportVectorData(:, 2), 'hg', 'MarkerFaceColor','g');
            title('prototype svm');
            colormap(bluewhitered);
            colorbar;
        end
        
        function prediction = predict(this, xTest)
            K = this.kModel.compute(this.supportVectorData, xTest); %%   note: has dimensions [m, n]
            m = size(K, 2);
            prediction = ([ones(1, m); K]' * this.prunedAlphas);
        end
    end
    
    
end

