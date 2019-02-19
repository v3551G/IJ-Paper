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
                %Kc = omega - 2 * onesM * omega + onesM * omega * onesM;
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
            figure; scatter(x(:, 1), x(:, 2), [], weights, 'filled'); grid on;
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
            initialSubsetMask(hIndices(1:ceil(h * n)))=true;
        end 
        
       
       function cstepmask = kernelCSteps(this, x, cstepmask, hInitial, hCstep, z)                       
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
                        display(['Confergence at iteration ' mat2str(iteration)]);
                        break;
                    end
                end
            end
            
              figure; scatter(x(:, 1), x(:, 2), [], smd, 'filled'); grid on;
            colorbar;
            title('obv univ. SDO');
            
            
                scores = (e.^2)./ repmat(Ld', size(e, 1), 1);                                        
                sdo = (scores - repmat(mean(scores(cstepmask, :)), size(scores, 1), 1)) ./ repmat(std(scores(cstepmask, :)), size(scores, 1), 1);
                sdo = max(abs(sdo), [], 2);
                hmask = sdo;
                %hmask = 1 - (sdo ./ max(sdo)); % zeros(size(sdo));
                %hmask(refinedSubset) = 1;
                figure; scatter(x(:, 1), x(:, 2), [], max(sdo.^2, [], 2), 'filled'); colorbar;
                weights = hmask;
                figure; scatter(x(:, 1), x(:, 2), [], weights, 'filled'); grid on;
                colorbar;
                title('obv kernel PCA');
            
       end
    end
    
    methods (Access = public)        
        function this = rLSSVM(kModel, pModel)
            this.kModel = kModel;
            this.pModel = pModel;
        end
        
        function train(this, dModel)
            %%%%    Process +1 class
            this.trainSingleClass(dModel.x(dModel.y>0, :), dModel.z(dModel.y>0, :));
            %%%%    Process -1 class
            this.trainSingleClass(dModel.x(dModel.y<0, :), dModel.z(dModel.y<0, :));
        end
        
        function trainSingleClass(this, x, z)
            
            hInitial = 0.5;
            hCstep = 0.60;
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%     Step 1
            %%%
            initisalSubset  = this.spatialMedian(x, hInitial);
            
            figure; 
            plot(x(initisalSubset, 1), x(initisalSubset, 2), '.g'); hold on;
            plot(x(~initisalSubset, 1), x(~initisalSubset, 2), '.r');
            title('Before');
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%     Step 2
            %%%
            refinedSubset = this.kernelCSteps(x, initisalSubset, hInitial, hCstep, z);
            
            figure; 
            plot(x(refinedSubset, 1), x(refinedSubset, 2), '.g'); hold on;
            plot(x(~refinedSubset, 1), x(~refinedSubset, 2), '.r');
            title('After');
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%     Step 3
            %%%            
            weights = this.sdoWeights(x, refinedSubset);
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%     Step 4
            %%%            
            lssvmModel = this.trainWeightedLSSVM(weights);
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%     Step 5
            %%%            
            prunedLsSvmModel = this.prune(lssvmModel);
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%     Step 6
            %%%            
            %efficency = prunedLsSvmModel.classify(...);
            
        end
    end
    
    
    
    
end

