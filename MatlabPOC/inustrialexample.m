
    clear;
    clc;
    close all;
    
    gp = csvread('datasets\if_cascajo0_32x450x450.csv');
    bp = csvread('datasets\if_molds0_32x450x450.csv');
    
    x=mean(gp, 2);
    xmask = (x<0.25);
    xroi = reshape(x, [450, 450]);
    xroi = xroi(:, 1:100);    
    
    y=mean(bp, 2);
    ymask = (y<0.25);
    bp(~ymask, :)
    
    yroi = reshape(y, [450, 450]);
    yroi = yroi(:, 1:100);
    
    figure; pcolor(xroi); shading flat;
    figure; pcolor(yroi); shading flat;
    
    csvwrite('gp.csv', gp(~xmask, :));
    csvwrite('maskx.csv', xmask(:, 1:100));
    csvwrite('bp.csv', bp(~ymask, :));
    csvwrite('masky.csv', ymask(:, 1:100));
    
    
    