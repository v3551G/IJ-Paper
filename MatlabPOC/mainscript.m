    close all;
    clear;
    clc;
    
    C = 1;
    
    rng default;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Kernel used
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
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Benchmark our classifier on unseen data
    %misclass = ourSvm.train(dModel); %, pModel);
    
    [rr, cc] = meshgrid(-3:0.05:3);    
    output = ourSvm.predict([rr(:), cc(:)]);    
    z=reshape(output, size(rr)); 
    figure; 
    contourf(rr, cc, z); hold on;
    plot(dModel.x(:, 1), dModel.x(:, 2), '.');
    colormap(bluewhitered);
    colorbar;
    
    
    
    
    
    
    
    