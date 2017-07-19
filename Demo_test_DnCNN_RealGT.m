clear
GT_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2017\cc_Results\Real_ccnoise_denoised_part\';
GT_fpath = fullfile(GT_Original_image_dir, '*mean.png');
TT_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2017\cc_Results\Real_ccnoise_denoised_part\';
TT_fpath = fullfile(TT_Original_image_dir, '*real.png');
% GT_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2017\cc_Results\Real_MeanImage\';
% GT_fpath = fullfile(GT_Original_image_dir, '*.png');
% TT_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2017\cc_Results\Real_NoisyImage\';
% TT_fpath = fullfile(TT_Original_image_dir, '*.png');
% GT_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2017\our_Results\Real_MeanImage\';
% GT_fpath = fullfile(GT_Original_image_dir, '*.JPG');
% TT_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2017\our_Results\Real_NoisyImage\';
% TT_fpath = fullfile(TT_Original_image_dir, '*.JPG');
% GT_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2017\1_Results\Real_NoisyImage\';
% GT_fpath = fullfile(GT_Original_image_dir, '*.png');
% TT_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2017\1_Results\Real_NoisyImage\';
% TT_fpath = fullfile(TT_Original_image_dir, '*.png');
GT_im_dir  = dir(GT_fpath);
TT_im_dir  = dir(TT_fpath);
im_num = length(TT_im_dir);

%%% This is the testing code demo for color image (Gaussian) denoising.
%%% The model is trained with 1) noise levels in [0 55]; 2) 432 training images.

addpath('utilities');

folderTest  = TT_Original_image_dir;

method           =  'DnCNN';
%% write image directory
write_sRGB_dir = ['C:/Users/csjunxu/Desktop/CVPR2017/cc_Results/'];
if ~isdir(write_sRGB_dir)
    mkdir(write_sRGB_dir)
end
folderModel = 'C:\Users\csjunxu\Desktop\JunXu\Paper\Image Video Denoising\DnCNN-master\model';
showResult  = 1;
useGPU      = 0;
pauseTime   = 1;

for noiseSigma  = 5  %%% image noise level
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
    %     ext         =  {'*.jpg','*.png','*.bmp'};
    ext         =  {'.JPG','.png'};
    NoisyfilePaths = [];
    meanfilePaths = [];
    for i = 1 : length(ext)
        NoisyfilePaths = cat(1,NoisyfilePaths, dir([folderTest '*real' ext{i}]));
        meanfilePaths = cat(1,meanfilePaths, dir([folderTest '*mean' ext{i}]));
    end
    %     for i = 1 : length(ext)
    %         NoisyfilePaths = cat(1,NoisyfilePaths, dir([folderTest '\*' ext{i}]));
    %         meanfilePaths = cat(1,meanfilePaths, dir([folderTest '\*' ext{i}]));
    %     end
    
    %%% PSNR and SSIM
    % PSNRs = zeros(1,length(filePaths));
    % SSIMs = zeros(1,length(filePaths));
    PSNRs = zeros(1,length(NoisyfilePaths));
    SSIMs = zeros(1,length(NoisyfilePaths));
    for i = 1:length(NoisyfilePaths)
        
        %%% read current image
        label = imread([folderTest '\' meanfilePaths(i).name]);
        %         label = imread([folderTest meanfilePaths(i).name]);
        [~,nameCur,extCur] = fileparts(meanfilePaths(i).name);
        label = im2double(label);
        tic
        %     %%% add Gaussian noise
        %     randn('seed',0);
        %     input = single(label + noiseSigma/255*randn(size(label)));
        input = imread([folderTest '\' NoisyfilePaths(i).name]);
        %         input = imread([folderTest NoisyfilePaths(i).name]);
        input = im2double(input);
        
        %%% convert to GPU
        if useGPU
            input = gpuArray(input);
        end
        
        %     res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test');
        res = simplenn_matlab(net, input); %%% use this if you did not install matconvnet.
        output = input - res(end).x;
        toc
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
        imwrite(im2uint8(output), [write_sRGB_dir 'Real_' method '/' method '_our100_' NoisyfilePaths(i).name]);
    end
    mPSNR = mean(PSNRs);
    mSSIM = mean(SSIMs);
    disp(mPSNR);
    disp(mSSIM);
    name = sprintf(['C:/Users/csjunxu/Desktop/CVPR2017/our_Results/', method, '_our100_' num2str(noiseSigma) '.mat']);
    save(name,'noiseSigma','mSSIM','mPSNR','PSNRs','SSIMs');
end
