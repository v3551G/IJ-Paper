

    %close all;
    clear;
    clc;
    
    rng default;
    
    pModel = EntropyBasedPruning();
    kModel = RbfKernel(10);
    
    dModel = YinYangDataModel(500, 0.1);         
    
    ourSvm = rLSSVM(kModel, pModel);    
    ourSvm.train(dModel);
    %misclass = ourSvm.train(dModel); %, pModel);
    
    
    
    
    
    
    
    
    
    