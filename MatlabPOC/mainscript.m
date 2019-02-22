    
    close all;
    clear all; 
    clc;
    
    %rng default;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Simulate the breaking of an svm in the linear case
    clear;
    
    C = 1e4;
    
    kModel = LinKernel();
    pModel = DPPBasedPruning(100, kModel);
    dModel = NormalDataModel(1500, 0.10);      
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
    
    C = 121;
    
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
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Wine classification test
    
    dModel = RedwineDataModel(1000, 0);    
    ourSvm = rLSSVM(kModel, pModel);    
    ourSvm.train(dModel, C);
    
    testdata = csvread('datasets\winequality-red.csv');    
    testLabels = zeros(size(testdata, 1), 1);
    mask = testdata(:, 12)>6;
    testLabels(mask) = +1; 
    testLabels(~mask) = -1;
    testdata = testdata(:, 1:11);
    
    response = sign(ourSvm.predict(testdata, dModel));
    misclass = sum(response~=testLabels) ./ numel(testLabels)
    
    