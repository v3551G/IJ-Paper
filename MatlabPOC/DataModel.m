classdef DataModel < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        %%% The bivariate data
        x;  
        %%% The labels [+1, -1]
        y;
        %%% The labels with the outliers marked as 0 [+1, 0, -1]
        z;
        %%% Use robust or naive standarization?
        robStandarization = true;
        %%% Keep normalization parameters so we can normalize the testdata
        mu;
        sigma;
    end
    
    methods
        function this = DataModel(n, p, eps)
            [ this.x, this.y, this.z ] = this.generateDataModel(n,p, eps);            
            [n, p] = size(this.x);
         
            if (this.robStandarization)
                %%%%%   Standarize the incoming dataset with the univariate MCD
                %%%%%   estimator            
                this.mu = nan(1, p);
                this.sigma = nan(1, p);            
                for featureIndex=1:p            
                    [tmcd,smcd] = unimcd(this.x(:, featureIndex), ceil(n*0.5));
                    this.mu(featureIndex) = tmcd;
                    this.sigma(featureIndex) = smcd;
                    assert(smcd>1e-10);
                end
            else
                %%%%%   Naive standarization
                this.mu = mean(this.x);
                this.sigma = std(this.x);
            end
            
            this.x = this.normalize(this.x);
        end
        
        function result = normalize(this, x)            
            n = size(x, 1);
            result = (x - repmat(this.mu, n, 1)) ./ repmat(this.sigma, n, 1);
        end
        
        function fh = plot(this)
            fh = figure(); 
            plot(this.x(this.z>0, 1), this.x(this.z>0, 2), '.g'); hold on;
            plot(this.x(this.z<0, 1), this.x(this.z<0, 2), '.b');
            plot(this.x(this.z==0, 1), this.x(this.z==0, 2), '.r');
            %grid on;
            set(fh, 'Color', 'w');
            %title('Generated data model ');
        end
    end
    
    methods (Abstract, Access = protected)
        generateDataModel(this, n, p, eps);
    end
    
end

