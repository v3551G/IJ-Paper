

    close all;
    clear;
    clc;
    C = 1;
    
    rng default;
    
    pModel = EntropyBasedPruning();
    kModel = RbfKernel(10);
    
    dModel = YinYangDataModel(500, 0.1);         
    
    ourSvm = rLSSVM(kModel, pModel);    
    ourSvm.train(dModel, C);
    %misclass = ourSvm.train(dModel); %, pModel);
    
    
    
    
    
    
    
    
    
    