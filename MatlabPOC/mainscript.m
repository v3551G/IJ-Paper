    
    close all;
    clear all; 
    clc;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Simulate the breaking of an svm in the linear case
    clear;
    rng default;
    
    C = 1e4;
    
    kModel = LinKernel();
    pModel = DPPBasedPruning(8, kModel);
    dModel = NormalDataModel(1500, 0.10);      
    fh=dModel.plot();    
    
    ourSvm = rLSSVM(kModel, pModel);    
    ourSvm.train(dModel, C);
    ourSvm.plot(dModel);
    
    lsSvm = LSSVM(kModel);
    lsSvm.train(dModel, C);
    lsSvm.plot(dModel);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Kernel used    
    clear;
    close all;
    rng default;
    
    C = 120;
    
    kModel = RbfKernel(0.5);
    %kModel = LinKernel();
    %kModel = PolyKernel(5);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Pruning strategy used
    %pModel = EntropyBasedPruning(10, kModel);
    pModel = DPPBasedPruning(10, kModel);
    
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
    clear;
    
    C = 120;
    
    kModel = RbfKernel(0.05);
    pModel = DPPBasedPruning(4, kModel);
    
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
    misclass = sum(response~=testLabels) ./ numel(testLabels);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Industrial dataset example
    clear;
    
    C = 1e4;
    
    kModel = RbfKernel(1);
    pModel = DPPBasedPruning(8, kModel);
    dModel = MoldDataModel(500, 0.10);      
    %fh=dModel.plot();    
    
    ourSvm = rLSSVM(kModel, pModel);    
    ourSvm.train(dModel, C);
    xResponse = ourSvm.predict(csvread('datasets\gp.csv'), dModel);
    xResponseImage = csvread('datasets\maskx.csv');
    
    z = nan(size(xResponseImage));
    z(xResponseImage>0) = xResponse;
    figure; pcolor(reshape(z, [450, 450])); shading flat;
    
    yResponse = ourSvm.predict(csvread('datasets\bp.csv'));
    
    
    lsSvm = LSSVM(kModel);
    lsSvm.train(dModel, C);
   
    
    