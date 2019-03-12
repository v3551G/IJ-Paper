classdef rLSSVM < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (Access = private)
        kModel;
        pModel;
        supportVectorData;
        supportVectorLabels;
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
            assert(n>0);
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
            hRange = hInitial:0.05:hCstep;
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
                %fh = figure(); plot(x(:, 1), x(:, 2), '.'); hold on; plot(x(cstepmask, 1), x(cstepmask, 2), 'p');
                %xlim([-4, 4]); ylim([-4, 4]); set(fh, 'Color', 'w'); 
                %export_fig('c2output.pdf');
            end
            
            %%%%%%%%%   Convert Hard rejection to outlying weights
            %%%
            %%% Stahel-Donoho with cellwise weights -> Hubert cost
            %%% function, eq. 2, pg 3  
            %%% Note that we're not using Stahel-Donoho weights but rather
            %%% the cost function based on mahalanobis distances
            %%% *edit: Fuck it, we're using hard rejection, Iwein
            
            scores = smd;
            sdo = max(abs( (scores - repmat(mean(scores(cstepmask, :)), size(scores, 1), 1)) ./ repmat(std(scores(cstepmask, :)), size(scores, 1), 1)), [], 2);            
            p = sum(lmask);            
            c = max(chi2inv(0.975, p), max(sdo(cstepmask)));            
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
            hInitial = 0.20;
            hCstep = 0.75;
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%    Process +1 class
            [w1, i1] = this.trainSingleClass(dModel.x(dModel.y>0, :), hInitial, hCstep);
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%    Process -1 class
            [w2, i2] = this.trainSingleClass(dModel.x(dModel.y<0, :), hInitial, hCstep);
            
            m1 = find(dModel.y>0);
            m2 = find(dModel.y<0);
            fh = figure(); 
            plot(dModel.x(:, 1), dModel.x(:, 2), '.', 'MarkerSize', 12, 'MarkerEdgeColor', [0,0,0]+0.9); hold on;
            plot(dModel.x(m1(i1), 1), dModel.x(m1(i1), 2), 'dm', 'MarkerSize', 4);  
            plot(dModel.x(m2(i2), 1), dModel.x(m2(i2), 2), 'db', 'MarkerSize', 4);  
            set(fh, 'Color', 'w');
            export_fig([ dModel.getfilename() '_sm.pdf']);
           
            fh = figure(); 
            plot(dModel.x(:, 1), dModel.x(:, 2), '.', 'MarkerSize', 12, 'MarkerEdgeColor', [0,0,0]+0.9); hold on;
            plot(dModel.x(m1(w1), 1), dModel.x(m1(w1), 2), 'dm', 'MarkerSize', 4);  
            plot(dModel.x(m2(w2), 1), dModel.x(m2(w2), 2), 'db', 'MarkerSize', 4);  
            set(fh, 'Color', 'w');
            export_fig([ dModel.getfilename() '_kc.pdf']);
            
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
            %%%
            %%%     TODO: Step 5b, balance alphas to compensate for class skew                        
            %%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            c1 = find(dModel.y>0 & weights & alphas>0);            
            c2 = find(dModel.y<0 & weights & alphas<0);
           
            fh = figure(); 
            plot(dModel.x(:, 1), dModel.x(:, 2), '.', 'MarkerSize', 12, 'MarkerEdgeColor', [0,0,0]+0.9); hold on;
            plot(dModel.x(c1, 1), dModel.x(c1, 2), 'dm', 'MarkerSize', 4);  
            plot(dModel.x(c2, 1), dModel.x(c2, 2), 'db', 'MarkerSize', 4);  
            set(fh, 'Color', 'w');
            export_fig([ dModel.getfilename() '_landm1.pdf']);
            
            m1 = c1(abs(alphas(c1))>median(abs(alphas(c1))));
            m2 = c2(abs(alphas(c2))>median(abs(alphas(c2))));
            
            fh = figure(); 
            plot(dModel.x(:, 1), dModel.x(:, 2), '.', 'MarkerSize', 12, 'MarkerEdgeColor', [0,0,0]+0.9); hold on;
            plot(dModel.x(m1, 1), dModel.x(m1, 2), 'dm', 'MarkerSize', 4);  
            plot(dModel.x(m2, 1), dModel.x(m2, 2), 'db', 'MarkerSize', 4);  
            set(fh, 'Color', 'w');
            export_fig([ dModel.getfilename() '_landm2.pdf']);
            
            
            svIndices1 = this.pModel.prune(dModel.x(c1, :), abs(alphas(c1)));
            svIndices2 = this.pModel.prune(dModel.x(c2, :), abs(alphas(c2)));
            
            fh = figure(); 
            plot(dModel.x(:, 1), dModel.x(:, 2), '.', 'MarkerSize', 12, 'MarkerEdgeColor',[0,0,0]+0.9); hold on;
            plot(dModel.x(c1(svIndices1), 1), dModel.x(c1(svIndices1), 2), 'dm', 'MarkerSize', 4, 'markerfacecolor', 'm');  
            plot(dModel.x(c2(svIndices2), 1), dModel.x(c2(svIndices2), 2), 'db', 'MarkerSize', 4, 'markerfacecolor', 'b');  
            set(fh, 'Color', 'w');
            export_fig([ dModel.getfilename() '_landm3.pdf']);
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%     Step 6, information transfer
            %%%            
            
            K = this.kModel.compute(dModel.x);
         
            supportVectorMask = false(size(dModel.y));
            supportVectorMask(c1(svIndices1)) = true;
            supportVectorMask(c2(svIndices2)) = true;
            
            c11 = dModel.y>0 & weights;
            c21 = dModel.y<0 & weights;            
            notsupportVectorMask = false(size(dModel.y));
            notsupportVectorMask(c11) = true;
            notsupportVectorMask(c21) = true;
            notsupportVectorMask(supportVectorMask) = false;
            
            delta_alpha = pinv(K(:,supportVectorMask))*K(:,notsupportVectorMask)*alphas(notsupportVectorMask, :);
            svAlphas =  alphas(supportVectorMask, :) + delta_alpha;
            this.supportVectorData = dModel.x(supportVectorMask,:);
            this.supportVectorLabels = dModel.y(supportVectorMask);
            this.prunedAlphas = [solution(1); svAlphas];
      
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%    That's all, folks!            
        end
        
        function [weights, initisalSubset]  = trainSingleClass(this, x, hInitial, hCstep)
            
            initisalSubset  = this.spatialMedian(x, hInitial);            
            weights = this.kernelCSteps(x, initisalSubset, hInitial, hCstep);            
        end
    
        function plot(this, dModel)
            [rr, cc] = meshgrid(-4:0.01:4);             
            output = sign(this.predict([rr(:), cc(:)], dModel));    
            z=reshape(output, size(rr)); 
            
            m1=this.supportVectorLabels>0;            
            fh = figure(); 
            contour(rr, cc, z, [0 0]); hold on;                        
            plot(dModel.x(:, 1), dModel.x(:, 2), '.', 'MarkerSize', 12, 'MarkerEdgeColor',[0,0,0]+0.9); hold on;
            plot(this.supportVectorData(m1, 1), this.supportVectorData(m1, 2), 'dm', 'MarkerSize', 12, 'markerfacecolor', 'm');  
            plot(this.supportVectorData(~m1, 1), this.supportVectorData(~m1, 2), 'db', 'MarkerSize', 12, 'markerfacecolor', 'b');  
            set(fh, 'Color', 'w');
            export_fig([ dModel.getfilename() '_hyperplane_robsvm.pdf']);
        end
        
        function prediction = predict(this, xTest, dModel)
            K = this.kModel.compute(this.supportVectorData, dModel.normalize(xTest)); %%   note: has dimensions [m, n]
            m = size(K, 2);
            prediction = ([ones(1, m); K]' * this.prunedAlphas);
        end
    end
    
    
end

