classdef SR_LSSVM < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (Access = private)
        kModel;
        pModel;
        supportVectorData;
        prunedAlphas;
    end
    
    methods (Access = public)        
        function this = SR_LSSVM(kModel, pModel)
            this.kModel = kModel;
            this.pModel = pModel;
        end
        
        function train(this, dModel, C)
            
            lam = C;
            
            [BS,P]=PCP_LSSVM(dModel.x, size(dModel.x, 1), kerhp, subsetsize, 10e-2);
            m=size(BS,2);
            s=zeros(n,1);olds=s+0.02;
            
            t=1;            
            p=10^4;
            
            tmp=sum(P);
            L=chol(lam*eye(m)+P'*P-tmp'*(tmp/n));
            
            while norm(s-olds)>rou && t<=itermax
                y=train_label-s;    
                tmp1 =L\(L'\(P'*(y-sum(y)/n)));
                a = P(BS,:)'\tmp1;
                b = (sum(y)-tmp*tmp1)/n;
                olds=s;
                r1=train_label-P*(P(BS,:)'*a)-b;
                s=r1.*min(1,exp(p*(r1.^2-tau.^2)))./(1+exp(-p*abs(r1.^2-tau.^2)));
                t=t+1;    
            end
            trtime=toc;
            
            this.prunedAlphas = [];
            this.supportVectorData = dModel.x(bs, :);
        end
        
        function prediction = predict(this, xTest, dModel)
            K = this.kModel.compute(this.supportVectorData, dModel.normalize(xTest)); %%   note: has dimensions [m, n]
            m = size(K, 2);
            prediction = ([ones(1, m); K]' * this.prunedAlphas);
        end
    end
    
    
end

