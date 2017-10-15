clear;
%%% This is the testing code demo for color image (Gaussian) denoising.
%%% The model is trained with 1) noise levels in [0 55]; 2) 432 training images.
addpath('utilities');

Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2018 Denoising\dnd_2017\images_srgb\';
fpath = fullfile(Original_image_dir, '*.mat');
im_dir  = dir(fpath);
im_num = length(im_dir);
load 'C:\Users\csjunxu\Desktop\CVPR2018 Denoising\dnd_2017\info.mat';

method = 'DnCNN';
% write image directory
write_MAT_dir = ['C:/Users/csjunxu/Desktop/CVPR2018 Denoising/dnd_2017Results/'];
write_sRGB_dir = ['C:/Users/csjunxu/Desktop/CVPR2018 Denoising/dnd_2017Results/' method];
if ~isdir(write_sRGB_dir)
    mkdir(write_sRGB_dir)
end

folderTest  = Original_image_dir;

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
        net = vl_simplenn_move(net, 'gpu');
    end
    %%% read images
    ext         =  {'*.jpg','*.png','*.bmp','.mat'};
    NoisyfilePaths = [];
    meanfilePaths = [];
    for i = 1 : length(ext)
        NoisyfilePaths = cat(1,NoisyfilePaths, dir([folderTest '\*' ext{i}]));
        meanfilePaths = cat(1,meanfilePaths, dir([folderTest '\*' ext{i}]));
    end
    %%% PSNR and SSIM
    PSNR = zeros(1,length(NoisyfilePaths));
    SSIM = zeros(1,length(NoisyfilePaths));
    RunTime = [];
    for i = 1:im_num
        load(fullfile(Original_image_dir, im_dir(i).name));
        S = regexp(im_dir(i).name, '\.', 'split');
        [h,w,ch] = size(InoisySRGB);
        for j = 1:size(info(1).boundingboxes,1)
            time0 = clock;
            IMinname = [S{1} '_' num2str(j)];
            input = InoisySRGB(info(i).boundingboxes(j,1):info(i).boundingboxes(j,3),info(i).boundingboxes(j,2):info(i).boundingboxes(j,4),1:3);
            time0 = clock;
            %%% read current image
            label = input;
            [~,nameCur,extCur] = fileparts(meanfilePaths(i).name);
            
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
            RunTime = [RunTime etime(clock,time0)];
            fprintf('Total elapsed time = %f s\n', (etime(clock,time0)) );
            %%% calculate PSNR
            psnr_cur = csnr(im2uint8(label), im2uint8(output), 0, 0);
            ssim_cur = cal_ssim(im2uint8(label), im2uint8(output), 0, 0);
            PSNR(i) = psnr_cur;
            SSIM(i) = ssim_cur;
            fprintf('The final PSNR = %2.4f, SSIM = %2.4f. \n', PSNR(i), SSIM(i));
            imwrite(im2uint8(output), [write_sRGB_dir '/' method '_DND_' IMinname '.png']);
        end
        clear InoisySRGB;
    end
    mPSNR = mean(PSNR);
    mSSIM = mean(SSIM);
    mRunTime = mean(RunTime);
    matname = sprintf([write_MAT_dir method '_DND.mat']);
    save(matname,'mSSIM','mPSNR','PSNR','SSIM','RunTime','mRunTime');
end
