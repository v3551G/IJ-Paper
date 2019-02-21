    close all;
    clear;
    clc;
    
    C = 1;
    
    rng default;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Retrain or info transfer (per class?)
    %rModel = Retrain();                 
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Pruning strategy used
    %pModel = EntropyBasedPruning(10, RbfKernel(0.1));
    pModel = DPPBasedPruning(10, RbfKernel(1));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Kernel used
    kModel = RbfKernel(1);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Data model used
    dModel = YinYangDataModel(1500, 0.1);         
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Build our classifier
    ourSvm = rLSSVM(kModel, pModel);    
    ourSvm.train(dModel, C);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Benchmark our classifier on unseen data
    %misclass = ourSvm.train(dModel); %, pModel);
    
    
    
    
    
    
    
    
    
    