    
    close all;
    clear; 
    clc;
    
    %rng default;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Simulate the breaking of an svm in the linear case
    clear;
    
    C = 120;
    
    kModel = LinKernel();
    pModel = DPPBasedPruning(100, kModel);
    dModel = NormalDataModel(1500, 0.1);      
    dModel.plot();    
    
    ourSvm = rLSSVM(kModel, pModel);    
    ourSvm.train(dModel, C);
    ourSvm.plot(dModel);
    
    lsSvm = LSSVM(kModel);
    lsSvm.train(dModel, C);
    lsSvm.plot(dModel);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Kernel used    
    clear;
    
    C = 120;
    
    kModel = RbfKernel(0.5);
    %kModel = LinKernel();
    %kModel = PolyKernel(5);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Pruning strategy used
    %pModel = EntropyBasedPruning(10, kModel);
    pModel = DPPBasedPruning(100, kModel);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Data model used
    dModel = YinYangDataModel(1500, 0.1);         
    dModel.plot();
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Build our classifier
    ourSvm = rLSSVM(kModel, pModel);    
    ourSvm.train(dModel, C);
    ourSvm.plot(dModel);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Benchmark our classifier on unseen data
    %misclass = ourSvm.train(dModel); %, pModel);
    
    lsSvm = LSSVM(kModel);
    lsSvm.train(dModel, C);
    lsSvm.plot(dModel);
    
    
    
    
    
    
    