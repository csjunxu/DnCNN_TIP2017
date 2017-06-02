
%%% This is the testing code demo for color image (Gaussian) denoising.
%%% The model is trained with 1) noise levels in [0 55]; 2) 432 training images.
%% read  image directory
% GT_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2017\our_Results\Real_MeanImage\';
% GT_fpath = fullfile(GT_Original_image_dir, '*.JPG');
% TT_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2017\our_Results\Real_NoisyImage\';
% TT_fpath = fullfile(TT_Original_image_dir, '*.JPG');
% GT_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2017\cc_Results\Real_ccnoise_denoised_part\';
% GT_fpath = fullfile(GT_Original_image_dir, '*mean.png');
% TT_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2017\cc_Results\Real_ccnoise_denoised_part\';
% TT_fpath = fullfile(TT_Original_image_dir, '*real.png');
% GT_im_dir  = dir(GT_fpath);
% TT_im_dir  = dir(TT_fpath);
% im_num = length(TT_im_dir);





clear;
addpath('utilities');
folderTest  = 'C:\Users\csjunxu\Desktop\JunXu\Datasets\kodak24\kodak_color\'; %%% test dataset

method           =  'DnCNNC';
%% write image directory
write_sRGB_dir = ['C:/Users/csjunxu/Desktop/ICCV2017/cc_Results/'];
if ~isdir(write_sRGB_dir)
    mkdir(write_sRGB_dir)
end
folderModel = 'model';
noiseSigma  = 45;  %%% image noise level
showResult  = 1;
useGPU      = 0;
pauseTime   = 1;

nSig = [40 20 30];
for noiseSigma  = 55  %%% image noise level
    %%% load blind Gaussian denoising model (color image)
    load(fullfile(folderModel,'GD_Color_Blind.mat')); %%% for sigma in [0,55]
    
    %%%
    % net = vl_simplenn_tidy(net);
    
    % for i = 1:size(net.layers,2)
    %     net.layers{i}.precious = 1;
    % end
    
    %%% move to gpu
    if useGPU
        net = vl_simplenn_move(net, 'gpu') ;
    end
    
    %%% read images
    ext         =  {'*.jpg','*.png','*.bmp'};
    meanfilePaths = [];
    for i = 1 : length(ext)
        meanfilePaths = cat(1,meanfilePaths, dir([folderTest ext{i}]));
    end
    
    %%% PSNR and SSIM
    % PSNRs = zeros(1,length(filePaths));
    % SSIMs = zeros(1,length(filePaths));
    PSNRs = zeros(1,length(meanfilePaths));
    SSIMs = zeros(1,length(meanfilePaths));
    for i = 1:length(meanfilePaths)
        
        %%% read current image
        label = imread([folderTest meanfilePaths(i).name]);
        [~,nameCur,extCur] = fileparts(meanfilePaths(i).name);
        label = im2double(label);
        S = regexp(meanfilePaths(i).name, '\.', 'split');
        IMname = S{1};
        %     %%% add Gaussian noise
        %     randn('seed',0);
        %     input = single(label + noiseSigma/255*randn(size(label)));
        input = zeros(size(label));
        [h, w, ch] = size(label);
        for c = 1:ch
            randn('seed',0);
            input(:, :, c) = label(:, :, c) + nSig(c)/255 * randn(size(label(:, :, c)));
        end
        
        
        %%% convert to GPU
        if useGPU
            input = gpuArray(input);
        end
        
        %     res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test');
        res = simplenn_matlab(net, input); %%% use this if you did not install matconvnet.
        output = input - res(end).x;
        
        %%% convert to CPU
        if useGPU
            output = gather(output);
            input  = gather(input);
        end
        
        %%% calculate PSNR
        %         [psnr_cur, ssim_cur] = Cal_PSNRSSIM(im2uint8(label),im2uint8(output),0,0);
        psnr_cur=csnr(im2uint8(label), im2uint8(output), 0, 0);
        ssim_cur = cal_ssim(im2uint8(label), im2uint8(output), 0, 0);
        %         if showResult
        %             imshow(cat(2,im2uint8(label),im2uint8(input),im2uint8(output)));
        %             title([NoisyfilePaths(i).name,'    ',num2str(psnr_cur,'%2.2f'),'dB','    ',num2str(ssim_cur,'%2.4f')])
        %             drawnow;
        %             pause(pauseTime)
        %         end
        PSNRs(i) = psnr_cur;
        SSIMs(i) = ssim_cur;
        fprintf('The final PSNR = %2.4f, SSIM = %2.4f. \n', PSNRs(i), SSIMs(i));
        imwrite(output, [write_sRGB_dir method '_nSig' num2str(nSig(1)) num2str(nSig(2)) num2str(nSig(3)) '_' IMname '.png']);
    end
    mPSNR = mean(PSNRs);
    mSSIM = mean(SSIMs);
    disp(mPSNR);
    disp(mSSIM);
    save(['C:/Users/csjunxu/Documents/GitHub/Weighted/OtherMethods/', method, '_nSig' num2str(nSig(1)) num2str(nSig(2)) num2str(nSig(3)) '_' num2str(noiseSigma) '.mat'],'noiseSigma','PSNRs','mPSNR','SSIMs','mSSIM');
end
