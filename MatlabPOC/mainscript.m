    
    clear;
    clc;
    
    C = 1;
    
    rng default;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Simulate the breaking of an svm in the linear case
    close all;
    
    kModel = LinKernel();
    dModel = NormalDataModel(1500, 0.1);         
    dModel.plot();    
    ourSvm = rLSSVM(kModel, pModel);    
    ourSvm.train(dModel, C);
    ourSvm.plot(dModel);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Kernel used
    
    close all;
    kModel = RbfKernel(0.5);
    %kModel = LinKernel();
    %kModel = PolyKernel(5);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Pruning strategy used
    %pModel = EntropyBasedPruning(10, kModel);
    pModel = DPPBasedPruning(20, kModel);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Data model used
    dModel = YinYangDataModel(1500, 0.1);         
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Build our classifier
    ourSvm = rLSSVM(kModel, pModel);    
    ourSvm.train(dModel, C);
    ourSvm.plot(dModel);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Benchmark our classifier on unseen data
    %misclass = ourSvm.train(dModel); %, pModel);
    
    
    
    
    
    
    
    
    