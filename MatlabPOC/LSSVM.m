classdef LSSVM < handle
    
    properties (Access = private)
        kModel;
        supportVectorData;
        prunedAlphas;
        supportVectorLabels;
    end
    
    methods (Access = public)        
        function this = LSSVM(kModel)
            this.kModel = kModel;
        end
        
        function train(this, dModel, C)
            yTrain = dModel.y;
            this.supportVectorData = dModel.x;  
            this.supportVectorLabels = dModel.y;
            n = numel(yTrain);
            K = this.kModel.compute(this.supportVectorData, this.supportVectorData);
            upper = [0, yTrain' ];
            lower = [yTrain , (yTrain * yTrain') .* K  + (1 ./ C) * eye(n,n) ];
            right = [0; ones(n, 1)];                            
            this.prunedAlphas = ([ upper; lower ] \ right) .* [1; yTrain];
        end
        
        function plot(this, dModel)
            [rr, cc] = meshgrid(-4:0.01:4);             
            output = sign(this.predict([rr(:), cc(:)], dModel));    
            z=reshape(output, size(rr)); 
            
            m1=this.supportVectorLabels>0;            
            fh = figure(); 
            contour(rr, cc, z, [0 0]); hold on;                        
            %plot(dModel.x(:, 1), dModel.x(:, 2), '.', 'MarkerSize', 12, 'MarkerEdgeColor',[0,0,0]+0.9); hold on;
            plot(this.supportVectorData(m1, 1), this.supportVectorData(m1, 2), '.m', 'MarkerSize', 4, 'markerfacecolor', 'm');  
            plot(this.supportVectorData(~m1, 1), this.supportVectorData(~m1, 2), '.b', 'MarkerSize', 4, 'markerfacecolor', 'b');  
            set(fh, 'Color', 'w');
            export_fig([ dModel.getfilename() '_hyperplane_lssvm.pdf']);
        end
        
        function prediction = predict(this, xTest, dModel)
            K = this.kModel.compute(this.supportVectorData, dModel.normalize(xTest)); %%   note: has dimensions [m, n]
            m = size(K, 2);
            prediction = ([ones(1, m); K]' * this.prunedAlphas);
        end
    end
    
    
end

