classdef FS_LSSVM < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (Access = private)
        kModel;
        pModel;
        supportVectorData;
        prunedAlphas;
    end
    
    methods (Access = public)        
        function this = FS_LSSVM(kModel, pModel)
            this.kModel = kModel;
            this.pModel = pModel;
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
        
        function train(this, dModel, C)
            svIndices1 = this.pModel.prune(dModel.x);            
            solution = this.trainWeightedLSSVM(dModel.x(svIndices1, :), dModel.y(svIndices1), C);
            this.prunedAlphas = solution;
            this.supportVectorData = dModel.x(svIndices1, :);
        end
        
        function prediction = predict(this, xTest, dModel)
            K = this.kModel.compute(this.supportVectorData, dModel.normalize(xTest)); %%   note: has dimensions [m, n]
            m = size(K, 2);
            prediction = ([ones(1, m); K]' * this.prunedAlphas);
        end
    end
    
    
end

