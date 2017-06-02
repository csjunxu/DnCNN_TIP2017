
%%% This is the testing demo for gray image (Gaussian) denoising.
%%% Training data: 400 images of size 180X180


clear; clc;
addpath('utilities');

Original_image_dir  =    'C:\Users\csjunxu\Desktop\JunXu\Datasets\kodak24\kodak_color\';
Sdir = regexp(Original_image_dir, '\', 'split');
fpath = fullfile(Original_image_dir, '*.png');
im_dir  = dir(fpath);
im_num = length(im_dir);
nSig = [40 20 30];


%% write image directory
write_sRGB_dir = ['C:/Users/csjunxu/Desktop/ICCV2017/24images/'];
if ~isdir(write_sRGB_dir)
    mkdir(write_sRGB_dir)
end


%% load [specific] Gaussian denoising model
folderModel = 'C:\Users\csjunxu\Desktop\JunXu\Paper\Image Video Denoising\DnCNN-master\model\';
showResult  = 0;
useGPU      = 0;
pauseTime   = 1;


%% PSNR and SSIM
PSNR = zeros(1,im_num);
SSIM = zeros(1,im_num);

for i = 1:im_num
    %% read images
    label = imread(fullfile(Original_image_dir,im_dir(i).name));
    S = regexp(im_dir(i).name, '\.', 'split');
    label = im2double(label);
    
    [h, w, ch] = size(label);
    input = zeros(size(label));
    output = zeros(size(label));
    for c = 1:ch
        randn('seed',0);
        input(:, :, c) = label(:, :, c) + nSig(c)/255 * randn(size(label(:, :, c)));
    end
    fprintf('%s :\n', im_dir(i).name);
    PSNR =   csnr( input*255, label*255, 0, 0 );
    SSIM      =  cal_ssim( input*255, label*255, 0, 0 );
    fprintf('The initial value of PSNR = %2.4f, SSIM = %2.4f \n', PSNR,SSIM);
            tic
    for c = 1:ch
        
        modelSigma  = min(75,max(10,round(nSig(c)/5)*5)); %%% model noise level
        load(fullfile(folderModel,'specifics',['sigma=',num2str(modelSigma,'%02d'),'.mat']));
        
        
        %% load [blind] Gaussian denoising model %%% for sigma in [0,55]
        
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

        %% convert to GPU
        if useGPU
            input(:, :, c) = gpuArray(input(:, :, c));
        end
        res = simplenn_matlab(net, input(:, :, c));
        %     res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test');
        %res = simplenn_matlab(net, input); %%% use this if you did not install matconvnet.
        output(:, :, c)  = input(:, :, c) - res(end).x;
    end
          toc
    %% convert to CPU
    if useGPU
        output = gather(output);
        input  = gather(input);
    end
    PSNRs(i)=csnr(label*255, output*255, 0, 0);
    SSIMs(i) = cal_ssim(label*255, output*255, 0, 0);
    
    fprintf('%s : PSNR = %2.4f, SSIM = %2.4f \n', im_dir(i).name, PSNRs(i), SSIMs(i)  );
    %         imname = sprintf('C:/Users/csjunxu/Desktop/NIPS2017/W3Results/DnCNN/DnCNN_nSig%d_%s', nSig, im_dir(i).name);
    %         imwrite(output, imname);
end
mPSNR=mean(PSNRs);
mSSIM=mean(SSIMs);
fprintf('The average PSNR = %2.4f, SSIM = %2.4f. \n', mPSNR,mSSIM);
name = sprintf([write_sRGB_dir 'DnCNNcw_nSig' num2str(nSig(1)) num2str(nSig(2)) num2str(nSig(3)) '.mat']);
save(name, 'nSig','PSNRs','SSIMs','mPSNR','mSSIM');

