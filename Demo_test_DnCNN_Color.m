clear
Original_image_dir  =    'C:\Users\csjunxu\Desktop\JunXu\Datasets\kodak24\kodak_color\';
fpath = fullfile(Original_image_dir, '*.png');
im_dir  = dir(fpath);
im_num = length(im_dir);

%%% This is the testing code demo for color image (Gaussian) denoising.
%%% The model is trained with 1) noise levels in [0 55]; 2) 432 training images.

addpath('utilities');
% folderTest  = 'C:\Users\csjunxu\Desktop\JunXu\Datasets\kodak24\kodak_color\'; %%% test dataset

folderTest  = 'C:\Users\csjunxu\Desktop\ICCV2017\24images\Noisy\';

method           =  'DnCNN';

%% write image directory
write_sRGB_dir = ['C:/Users/csjunxu/Desktop/ICCV2017/24images/'];
if ~isdir(write_sRGB_dir)
    mkdir(write_sRGB_dir)
end

nSig = [30 10 50];

folderModel = 'C:\Users\csjunxu\Desktop\JunXu\Paper\Image Video Denoising\DnCNN-master\model\';

showResult  = 1;
useGPU      = 0;
pauseTime   = 1;

for noiseSigma  = [25 30] %%% image noise level
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
    
    
    %%% PSNR and SSIM
    PSNRs = zeros(1, im_num);
    SSIMs = zeros(1, im_num);
    for i = 1:im_num
        
        %%% read current image
        label = double( imread(fullfile(Original_image_dir, im_dir(i).name)) )/255;
        %         label = im2double(label);
        tic
        %% add Gaussian noise
        [h, w, ch] = size(label);
        input = zeros(size(label));
        
        %         % global Gaussian
        %         randn('seed',0);
        %         input = label+ nSig(1)/255 * randn(size(label));
        %         each channel is Gaussian
        for c = 1:ch
            randn('seed',0);
            input(:, :, c) = label(:, :, c) + nSig(c)/255 * randn(size(label(:, :, c)));
        end
        
        fprintf('%s :\n', im_dir(i).name);
        PSNR =   csnr( input*255, label*255, 0, 0 );
        SSIM      =  cal_ssim( input*255, label*255, 0, 0 );
        fprintf('The initial value of PSNR = %2.4f, SSIM = %2.4f \n', PSNR,SSIM);
        %% convert to GPU
        if useGPU
            input = gpuArray(input);
        end
        
        %     res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test');
        res = simplenn_matlab(net, input); %% use this if you did not install matconvnet.
        output = input - res(end).x;
        toc
        %%% convert to CPU
        if useGPU
            output = gather(output);
            input  = gather(input);
        end
        
        %%% calculate PSNR
        %         [psnr_cur, ssim_cur] = Cal_PSNRSSIM(im2uint8(label),im2uint8(output),0,0);
        PSNRs(i)=csnr(label*255, output*255, 0, 0);
        SSIMs(i) = cal_ssim(label*255, output*255, 0, 0);
        %         if showResult
        %             imshow(cat(2,im2uint8(label),im2uint8(input),im2uint8(output)));
        %             title([NoisyfilePaths(i).name,'    ',num2str(psnr_cur,'%2.2f'),'dB','    ',num2str(ssim_cur,'%2.4f')])
        %             drawnow;
        %             pause(pauseTime)
        %         end
        fprintf('The final PSNR = %2.4f, SSIM = %2.4f. \n', PSNRs(i), SSIMs(i));
                imwrite(im2uint8(output), [write_sRGB_dir method '/' method '_nSig' num2str(nSig(1)) num2str(nSig(2)) num2str(nSig(3)) '_' im_dir(i).name]);
    end
    mPSNR = mean(PSNRs);
    mSSIM = mean(SSIMs);
    disp(mPSNR);
    disp(mSSIM);
    name = sprintf([write_sRGB_dir method '_nSig' num2str(nSig(1)) num2str(nSig(2)) num2str(nSig(3)) '.mat']);
    save(name,'nSig','mSSIM','mPSNR','PSNRs','SSIMs');
end



nSig = [5 30 15];

folderModel = 'C:\Users\csjunxu\Desktop\JunXu\Paper\Image Video Denoising\DnCNN-master\model\';

showResult  = 1;
useGPU      = 0;
pauseTime   = 1;

for noiseSigma  = [25 30] %%% image noise level
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
    
    
    %%% PSNR and SSIM
    PSNRs = zeros(1, im_num);
    SSIMs = zeros(1, im_num);
    for i = 1:im_num
        
        %%% read current image
        label = double( imread(fullfile(Original_image_dir, im_dir(i).name)) )/255;
        %         label = im2double(label);
        tic
        %% add Gaussian noise
        [h, w, ch] = size(label);
        input = zeros(size(label));
        
        %         % global Gaussian
        %         randn('seed',0);
        %         input = label+ nSig(1)/255 * randn(size(label));
        %         each channel is Gaussian
        for c = 1:ch
            randn('seed',0);
            input(:, :, c) = label(:, :, c) + nSig(c)/255 * randn(size(label(:, :, c)));
        end
        
        fprintf('%s :\n', im_dir(i).name);
        PSNR =   csnr( input*255, label*255, 0, 0 );
        SSIM      =  cal_ssim( input*255, label*255, 0, 0 );
        fprintf('The initial value of PSNR = %2.4f, SSIM = %2.4f \n', PSNR,SSIM);
        %% convert to GPU
        if useGPU
            input = gpuArray(input);
        end
        
        %     res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test');
        res = simplenn_matlab(net, input); %% use this if you did not install matconvnet.
        output = input - res(end).x;
        toc
        %%% convert to CPU
        if useGPU
            output = gather(output);
            input  = gather(input);
        end
        
        %%% calculate PSNR
        %         [psnr_cur, ssim_cur] = Cal_PSNRSSIM(im2uint8(label),im2uint8(output),0,0);
        PSNRs(i)=csnr(label*255, output*255, 0, 0);
        SSIMs(i) = cal_ssim(label*255, output*255, 0, 0);
        %         if showResult
        %             imshow(cat(2,im2uint8(label),im2uint8(input),im2uint8(output)));
        %             title([NoisyfilePaths(i).name,'    ',num2str(psnr_cur,'%2.2f'),'dB','    ',num2str(ssim_cur,'%2.4f')])
        %             drawnow;
        %             pause(pauseTime)
        %         end
        fprintf('The final PSNR = %2.4f, SSIM = %2.4f. \n', PSNRs(i), SSIMs(i));
        %         imwrite(im2uint8(output), [write_sRGB_dir method '/' method '_nSig' num2str(nSig(1)) num2str(nSig(2)) num2str(nSig(3)) '_' im_dir(i).name]);
    end
    mPSNR = mean(PSNRs);
    mSSIM = mean(SSIMs);
    disp(mPSNR);
    disp(mSSIM);
    name = sprintf([write_sRGB_dir method '_nSig' num2str(nSig(1)) num2str(nSig(2)) num2str(nSig(3)) '.mat']);
    save(name,'nSig','mSSIM','mPSNR','PSNRs','SSIMs');
end
