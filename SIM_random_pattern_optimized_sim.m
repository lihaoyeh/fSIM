% speckle structured illumination fluorescent microscopy
% Coded by Li-Hao Yeh 2015.08.07
% Last updated 2018.08.02

% clear all;
set(0,'DefaultFigureWindowStyle','docked');

F = @(x) fftshift(fft2(ifftshift(x)));
iF = @(x) fftshift(ifft2(ifftshift(x)));

% load res_speckle_shift.mat;

%% Coordinate assignment

[Ncrop,Mcrop,Nimg] = size(I_image);

% If the reconstructed resolution is smaller than Nyquist sampling, 
% apply higher factor


upsampling_factor = 1; 

N = Ncrop*upsampling_factor; M = Mcrop*upsampling_factor; ps = pscrop/upsampling_factor;

xh = (-M/2:(M/2-1)).*ps; yh = (-N/2:(N/2-1)).*ps;
fx = (-M/2:(M/2-1))./(ps*M); fy = (-N/2:(N/2-1))./(ps*N);
NAx = fx*lambda; NAy = fy*lambda;
[xhh,yhh] = meshgrid(xh,yh);
[fxx,fyy] = meshgrid(fx,fy);

fxx = ifftshift(fxx);
fyy = ifftshift(fyy);

N_bound_pad = 0;
Nc = N + 2*N_bound_pad;
Mc = M + 2*N_bound_pad;

fx_c = (-Mc/2:(Mc/2-1))./(ps*Mc); fy_c = (-Nc/2:(Nc/2-1))./(ps*Nc);
[fxx_c,fyy_c] = meshgrid(fx_c,fy_c);

fxx_c = ifftshift(fxx_c);
fyy_c = ifftshift(fyy_c);



%% Upsampling the data

F = @(x) fftshift(fft2(ifftshift(x)));
iF = @(x) fftshift(ifft2(ifftshift(x)));

I_image_up = zeros(N,M,Nimg);
Ic_image_up = zeros(N,M,Nimg);
for i = 1:Nimg
    I_image_up(:,:,i) = max(0,real(iF(padarray(F(max(0,I_image(:,:,i)-100)),[(N-Ncrop)/2,(M-Mcrop)/2]))));
    Ic_image_up(:,:,i) = max(0,real(iF(padarray(F(max(0,Ic_image(:,:,i)-100)),[(N-Ncrop)/2,(M-Mcrop)/2]))));
end


%% Image registration to get shift

xshift = zeros(Nimg,1);
yshift = zeros(Nimg,1);

for j = 1:Nimg
    if j == 1
        yshift(j) = 0;
        xshift(j) = 0;
    else
        [output, ~] = dftregistration(fft2(Ic_image_up(:,:,1)),fft2(Ic_image_up(:,:,j)),100);
        yshift(j) = (output(3)); 
        xshift(j) = (output(4));
    end
end


%% initialization

F = @(x) fft2(x);
iF = @(x) ifft2(x);

yshift_max = round(max(abs(yshift(:))));
xshift_max = round(max(abs(xshift(:))));


I_obj = gpuArray(padarray(mean(I_image_up,3),[N_bound_pad,N_bound_pad])); % object initial guess

I_forward = zeros(N,M,Nimg); % estimated intensity


I_p_whole = gpuArray(ones(Nc+2*yshift_max,Mc+2*xshift_max)); % pattern initialization
% I_p = zeros(Nc,Mc,Nimg); % croped estimated pattern

% set up coordinate for pattern to do shifting operation in Fourier space
Npp = Nc + 2*yshift_max;
Mpp = Mc + 2*xshift_max;
fxp = (-Mpp/2:(Mpp/2-1))./(ps*Mpp); fyp = (-Npp/2:(Npp/2-1))./(ps*Npp);
[fxxp,fyyp] = meshgrid(fxp,fyp); 
fxxp = gpuArray(ifftshift(fxxp));
fyyp = gpuArray(ifftshift(fyyp));


% for j = 1:Nimg
%     Ip_shift = max(0,real(iF(F(I_p_whole).*exp(1j*2*pi*ps*(fxxp.*xshift(j) + fyyp.*yshift(j))))));
%     I_p(:,:,j) = gather(Ip_shift(1+yshift_max:Nc+yshift_max,1+xshift_max:Mc+xshift_max));
% end

% compute transfer function for this experiment

Pupil_obj = zeros(Nc,Mc);
r_obj=(fxx_c.^2+fyy_c.^2).^(1/2);
Pupil_obj(find(r_obj<NA_obj/lambda))=1;
T_incoherent = abs(F(abs(iF(Pupil_obj)).^2));
T_incoherent = gpuArray(T_incoherent/max(T_incoherent(:)));

% support constraint for pattern update

NAs = 0.5;
Pupil_support = zeros(Npp,Mpp,'gpuArray');
Pupil_support(sqrt(fxxp.^2+fyyp.^2)<2*NAs/lambda) = 1;

% load([out_dir,'\Pupil.mat']);
% Pupil_obj = ifftshift(padarray(fftshift(Pupil_obj),[(N-Ncrop)/2,(M-Mcrop)/2]));

Pupil_obj_support = zeros(Nc,Mc);
Pupil_obj_support(sqrt(fxx_c.^2+fyy_c.^2)<2*(NAs+NA_obj)/lambda) = 1;
Pupil_obj_support = gpuArray(Pupil_obj_support);

Pupil_obj_f_2NA = zeros(Nc,Mc);
Pupil_obj_f_2NA(r_obj<2*NA_obj/lambda) = 1;
Pupil_obj_f_2NA = gpuArray(Pupil_obj_f_2NA);

% iteration number 

itr = 70;

% cost function

err = zeros(1,itr+1);

% calculate the initial cost function value

% for j = 1:Nimg
%     I_est = iF(T_incoherent.*F(gpuArray(I_p(:,:,j)).*I_obj));
%     I_forward(:,:,j) = gather(I_est(N_bound_pad+1:N_bound_pad+N,N_bound_pad+1:N_bound_pad+M));
% end

% err(1) = sum(sum(sum((I_image_up-I_forward).^2)));

%% Iterative algorithm

tic;
fprintf('| Iter  |   error    | Elapsed time (sec) |\n');
for i = 1:itr
    
    % Sequential update
    
    for j = 1:Nimg
        
        Ip_shift = max(0,real(iF(F(I_p_whole).*exp(1j*2*pi*ps*(fxxp.*xshift(j) + fyyp.*yshift(j))))));
        I_p_gpu = Ip_shift(1+yshift_max:Nc+yshift_max,1+xshift_max:Mc+xshift_max);
%         I_p(:,:,j) = gather(I_p_gpu);
        
        I_image_current = gpuArray(I_image_up(:,:,j));

        I_multi_f = F(I_p_gpu.*I_obj);
        I_est =  iF(T_incoherent.*I_multi_f);        
        I_diff = I_image_current - I_est(N_bound_pad+1:N_bound_pad+N,N_bound_pad+1:N_bound_pad+M);
        
        I_temp = iF(T_incoherent.*F(padarray(I_diff,[N_bound_pad,N_bound_pad])));

        grad_Iobj = -real(I_p_gpu.*I_temp);        
        grad_Ip = -real(iF(F(padarray(I_obj.*I_temp/max(I_obj(:))^2,[yshift_max,xshift_max],0)).*exp(-1j*2*pi*ps*(fxxp.*xshift(j) + fyyp.*yshift(j)))));            
        grad_OTF = -conj(I_multi_f).*F(I_temp).*Pupil_obj_f_2NA;
        
        
        I_obj = real(iF(F(I_obj - real(grad_Iobj/max(I_p_gpu(:))^2)).*Pupil_obj_support));
        I_p_whole = real(iF(F(I_p_whole -grad_Ip).*Pupil_support));
%         T_incoherent = T_incoherent - grad_OTF/max(abs(I_multi_f(:))).*abs(I_multi_f)./(abs(I_multi_f).^2 + 1e-3)/12;

        
        % shift estimate
        Ip_shift_fx = iF(F(I_p_whole).*(1j*2*pi*fxxp).*exp(1j*2*pi*ps*(fxxp.*xshift(j) + fyyp.*yshift(j))));
        Ip_shift_fy = iF(F(I_p_whole).*(1j*2*pi*fyyp).*exp(1j*2*pi*ps*(fxxp.*xshift(j) + fyyp.*yshift(j))));
        
        Ip_shift_fx = Ip_shift_fx(1+yshift_max:Nc+yshift_max,1+xshift_max:Mc+xshift_max);
        Ip_shift_fy = Ip_shift_fy(1+yshift_max:Nc+yshift_max,1+xshift_max:Mc+xshift_max);

        grad_xshift = -real(sum(sum(conj(I_temp).*I_obj.*Ip_shift_fx)));
        grad_yshift = -real(sum(sum(conj(I_temp).*I_obj.*Ip_shift_fy)));

        xshift(j) = xshift(j) - gather(grad_xshift/N/M/max(I_obj(:))^2);
        yshift(j) = yshift(j) - gather(grad_yshift/N/M/max(I_obj(:))^2);
        
        err(i+1) = err(i+1) + gather(sum(sum(abs(I_diff).^2)));

        
    end
    
    
    temp = I_obj;
    temp_Ip = I_p_whole;
    if i == 1           
        t = 1;
        
        I_obj = temp;
        tempp = temp;
        
        I_p_whole = temp_Ip;
        tempp_Ip = temp_Ip;
    else
        if (err(i) >= err(i-1))
            t = 1;
        
            I_obj = temp;
            tempp = temp;

            I_p_whole = temp_Ip;
            tempp_Ip = temp_Ip;
        else
            tp = t;
            t = (1+sqrt(1+4*tp^2))/2;

            I_obj = temp + (tp-1)*(temp - tempp)/t;
            tempp = temp;

            I_p_whole = temp_Ip + (tp-1)*(temp_Ip - tempp_Ip)/t;
            tempp_Ip = temp_Ip;
        end
    end
    
   
%     for j = 1:Nimg
%         I_est = iF(T_incoherent.*F(gpuArray(I_p(:,:,j)).*I_obj));
%         I_forward(:,:,j) = gather(I_est(N_bound_pad+1:N_bound_pad+N,N_bound_pad+1:N_bound_pad+M));
%     end
% 
%     err(i+1) = sum(sum(sum((I_image_up-I_forward).^2)));
    
    if mod(i,1) == 0
        fprintf('|  %2d   |  %.2e  |        %.2f        |\n', i, err(i+1),toc);
        figure(31);
        subplot(1,2,1),imagesc(I_obj,[0 max(I_obj(:))]);colormap gray;axis square;
        subplot(1,2,2),imagesc(I_p_whole,[0 max(I_p_whole(:))]);colormap gray;axis square;
%         figure(32);
%         subplot(1,2,1),plot(xshift,yshift,'bo');axis square;
%         subplot(1,2,2),imagesc(abs(fftshift(T_incoherent)));colormap jet;axis image;
        pause(0.001);
    end

end
