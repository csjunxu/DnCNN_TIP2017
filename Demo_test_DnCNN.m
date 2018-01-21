
%%% This is the testing demo for gray image (Gaussian) denoising.
%%% Training data: 400 images of size 180X180


clear; clc;
addpath('utilities');

Original_image_dir  =    'C:\Users\csjunxu\Desktop\Projects\WODL\20images\';
Sdir = regexp(Original_image_dir, '\', 'split');
fpath = fullfile(Original_image_dir, '*.png');
im_dir  = dir(fpath);
im_num = length(im_dir);

for nSig  = [20 40 60 80 100]  %%% image noise level
    %%% load [specific] Gaussian denoising model
    folderModel = 'model';
    showResult  = 0;
    useGPU      = 0;
    pauseTime   = 1;
    modelSigma  = min(75,max(10,round(nSig/5)*5)); %%% model noise level
    load(fullfile(folderModel,'specifics',['sigma=',num2str(modelSigma,'%02d'),'.mat']));
    
    %%% load [blind] Gaussian denoising model %%% for sigma in [0,55]
    
    % load(fullfile(folderModel,'GD_Gray_Blind.mat'));
    
    
    %%%
    % net = vl_simplenn_tidy(net);
    
    % for i = 1:size(net.layers,2)
    %     net.layers{i}.precious = 1;
    % end
    
    %%% move to gpu
    if useGPU
        net = vl_simplenn_move(net, 'gpu') ;
    end
    
    %%% PSNR and SSIM
    PSNR = zeros(1,im_num);
    SSIM = zeros(1,im_num);
    
    for i = 1:im_num
        
        %%% read images
        label = imread(fullfile(Original_image_dir,im_dir(i).name));
        S = regexp(im_dir(i).name, '\.', 'split');
        label = im2double(label);
        
        randn('seed',0);
        input = single(label + nSig/255*randn(size(label)));
        
        %%% convert to GPU
        if useGPU
            input = gpuArray(input);
        end
        res = simplenn_matlab(net, input);
        %     res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test');
        %res = simplenn_matlab(net, input); %%% use this if you did not install matconvnet.
        output = input - res(end).x;
        
        %%% convert to CPU
        if useGPU
            output = gather(output);
            input  = gather(input);
        end
        
        %%% calculate PSNR and SSIM
        [PSNRCur, SSIMCur] = Cal_PSNRSSIM(im2uint8(label),im2uint8(output),0,0);
        if showResult
            imshow(cat(2,im2uint8(label),im2uint8(input),im2uint8(output)));
            title([im_dir(i).name,'    ',num2str(PSNRCur,'%2.2f'),'dB','    ',num2str(SSIMCur,'%2.4f')])
            drawnow;
            pause(pauseTime)
        end
        PSNR(i) = PSNRCur;
        SSIM(i) = SSIMCur;
        fprintf('%s : PSNR = %2.4f, SSIM = %2.4f \n', im_dir(i).name, PSNR(i), SSIM(i)  );
        imname = sprintf('C:/Users/csjunxu/Desktop/NIPS2017/W3Results/DnCNN/DnCNN_nSig%d_%s', nSig, im_dir(i).name);
        imwrite(output, imname);
    end
    mPSNR=mean(PSNR);
    mSSIM=mean(SSIM);
    fprintf('The average PSNR = %2.4f, SSIM = %2.4f. \n', mPSNR,mSSIM);
    name = sprintf(['C:/Users/csjunxu/Desktop/NIPS2017/W3Results/DnCNN_nSig' num2str(nSig) '.mat']);
    save(name, 'nSig','PSNR','SSIM','mPSNR','mSSIM');
end
