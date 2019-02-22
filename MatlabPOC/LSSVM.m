classdef LSSVM < handle
    
    properties (Access = private)
        kModel;
        supportVectorData;
        prunedAlphas;
    end
    
    methods (Access = public)        
        function this = LSSVM(kModel)
            this.kModel = kModel;
        end
        
        function train(this, dModel, C)
            yTrain = dModel.y;
            this.supportVectorData = dModel.x;            
            n = numel(yTrain);
            K = this.kModel.compute(this.supportVectorData, this.supportVectorData);
            upper = [0, yTrain' ];
            lower = [yTrain , (yTrain * yTrain') .* K  + (1 ./ C) * eye(n,n) ];
            right = [0; ones(n, 1)];                            
            this.prunedAlphas = ([ upper; lower ] \ right) .* [1; yTrain];
        end
        
        function plot(this, dModel)
            [rr, cc] = meshgrid(-4:0.05:4);    
            output = sign(this.predict([rr(:), cc(:)], dModel));    
            z=reshape(output, size(rr)); 
            figure; 
            contourf(rr, cc, z, [0 0]); hold on;
            if (nargin==2)
                plot(dModel.x(:, 1), dModel.x(:, 2), '.');
            end
            if (size(dModel.x, 1)~=size(this.supportVectorData, 1))
                plot(this.supportVectorData(:, 1), this.supportVectorData(:, 2), 'hg', 'MarkerFaceColor','g');
            end
            colormap(bluewhitered);
            title('normal svm');
            %colorbar;
        end
        
        function prediction = predict(this, xTest, dModel)
            K = this.kModel.compute(this.supportVectorData, dModel.normalize(xTest)); %%   note: has dimensions [m, n]
            m = size(K, 2);
            prediction = ([ones(1, m); K]' * this.prunedAlphas);
        end
    end
    
    
end

