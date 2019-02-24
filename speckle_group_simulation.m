% Generate simulation data for shifted speckle illumination 
% Coded by Li-Hao Yeh 2016.08.13
% Last update 2017.02.22
% Add coherent simulation

clear all;
set(0,'DefaultFigureWindowStyle','docked');



F = @(x) fftshift(fft2(ifftshift(x)));
iF = @(x) fftshift(ifft2(ifftshift(x)));

I = double(imread('resolution.jpg'));

% I = I(:,:,2);
I = padarray(I(129:516,345:732,2),[6,6]);
% I = I(323:506,358:541,2);
% I = padarray(I(323:506,358:541,2),[30,30]);

I = I/max(I(:));

amp = (1-I)+0.5;
amp = amp/max(amp(:));
% amp = ones(size(amp));
ph = I;

T_c = amp.*exp(1j*ph);



%% --------Experiment Setup----------

lambda = 0.605; k=2*pi/lambda; % wavelength (micons) and wave number
mag = 40;
pscrop = 6.5/mag; % Pixels size (microns)
NA_obj = 0.1;
NAs = 0.4;

undersamp_factor = 1;
ps = pscrop/undersamp_factor;


[N,M] = size(I);

Ncrop = N/undersamp_factor;
Mcrop = M/undersamp_factor;

xh = (-M/2:(M/2-1)).*ps; yh = (-N/2:(N/2-1)).*ps;
fx = (-M/2:(M/2-1))./(ps*M); fy = (-N/2:(N/2-1))./(ps*N);
NAx = fx*lambda; NAy = fy*lambda;
[xhh,yhh] = meshgrid(xh,yh);
[fxx,fyy] = meshgrid(fx,fy);
n = xhh./ps;
m = yhh./ps;


%% Star generator
% [theta,rho] = cart2pol(xhh(:),yhh(:));
% 
% theta = reshape(theta,N,M);
% rho = reshape(rho,N,M);
% 
% I = 1+cos(40*theta);
% I = I(11:N-10,11:M-10);
% I = padarray(I,[10,10]);
% 
% figure;imagesc(I);colormap gray;axis image;axis off;

%% -------- Propagation kernel --------

xs = (-M*4/2:(M*4/2-1)).*ps; ys = (-N*4/2:(N*4/2-1)).*ps;
fxs = (-M*4/2:(M*4/2-1))./(ps*4*M); fys = (-N*4/2:(N*4/2-1))./(ps*4*N);
NAxs = fxs.*lambda; NAys = fys.*lambda;
[xss,yss] = meshgrid(xs,ys);
[fxxs,fyys] = meshgrid(fxs,fys);
NAxx = fxxs.*lambda;
NAyy = fyys.*lambda;


r_prop=(fxxs.^2+fyys.^2).^(1/2);
Pupil_prop = zeros(N*4,M*4);
Pupil_prop(find(r_prop<NAs/lambda))=1;


Pupil_2NA = zeros(N*4,M*4);
Pupil_2NA(find(r_prop<2*NAs/lambda))=1;

%% Pixel shift in group

N_pattern = 1;

N_shiftx = 25;
N_shifty = 25;

Nimg = N_shiftx*N_shifty*N_pattern;
N_shift = N_shiftx*N_shifty;

% pixel_step = [2;2;2;2;2;2;2;2;2;2];
pixel_step = 2*ones(N_pattern,1);
pixel_shift_stack = zeros(2,N_shift,N_pattern);


for i = 1:N_pattern
    pixel_shiftx = (-(N_shiftx-1)/2:(N_shiftx-1)/2).*pixel_step(i);
    pixel_shifty = (-(N_shifty-1)/2:(N_shifty-1)/2).*pixel_step(i);
%     pixel_shiftx = (-(N_shiftx)/2:(N_shiftx)/2-1).*pixel_step(i);
%     pixel_shifty = (-(N_shifty)/2:(N_shifty)/2-1).*pixel_step(i);
    [pixel_shiftyy,pixel_shiftxx] = meshgrid(pixel_shifty,pixel_shiftx);
    
    pixel_shift_stack(1,:,i) = pixel_shiftyy(:);
    pixel_shift_stack(2,:,i) = pixel_shiftxx(:);
end

%% Pattern generation with random phase mask

rng(49);

speckle_intensity = zeros(4*N,4*M,N_pattern);
speckle_field = zeros(4*N,4*M,N_pattern);

for i = 1:N_pattern
    random_mapf = exp(1j*rand(4*N,4*M)*100);
    
    temp = iF(random_mapf.*Pupil_prop);
%     temp = abs(iF(F(temp).*Pupil_2NA));
    speckle_field(:,:,i) = temp/max(abs(temp(:)));
    speckle_intensity(:,:,i) = abs(speckle_field(:,:,i)).^2;
end

speckle_intensity_crop = speckle_intensity(1.5*N+1:2.5*N,1.5*M+1:2.5*M,1);
speckle_intensity_cropf = F(speckle_intensity_crop);
% speckle_field = rand(4*N,4*N);

figure;imagesc(xh,yh,speckle_intensity_crop); colormap gray;axis square;
% % figure;imagesc(NAx,NAy,log10(abs(F(speckle_field_crop))),[1 5]);colormap jet;axis square;
figure;imagesc(NAx,NAy,log10(abs(speckle_intensity_cropf))/max(max(log10(abs(speckle_intensity_cropf)))),[0 1]); colormap jet; axis square;
hold on;circle(0,0,2*NAs);


%% Data generation

Pupil_obj = zeros(N,M);
r_obj=(fxx.^2+fyy.^2).^(1/2);
Pupil_obj(find(r_obj<NA_obj/lambda))=1;
T_incoherent = abs(F(abs(iF(Pupil_obj)).^2));
T_incoherent = T_incoherent/max(T_incoherent(:));

speckle_intensity_shift_crop = zeros(N,M,Nimg);
speckle_field_shift_crop = zeros(N,M,Nimg);
I_image = zeros(Ncrop,Mcrop,Nimg);
Ic_image = zeros(Ncrop,Mcrop,Nimg);

for i = 1:Nimg
    
    idx_ps = mod(i-1,N_shift)+1;
    idx_pt = floor((i-1)/N_shift)+1;
    
    speckle_intensity_shift_crop(:,:,i) = speckle_intensity(1.5*N+1+pixel_shift_stack(1,idx_ps,idx_pt):2.5*N+pixel_shift_stack(1,idx_ps,idx_pt),1.5*M+1+pixel_shift_stack(2,idx_ps,idx_pt):2.5*M+pixel_shift_stack(2,idx_ps,idx_pt),idx_pt);
    speckle_field_shift_crop(:,:,i) = speckle_field(1.5*N+1+pixel_shift_stack(1,idx_ps,idx_pt):2.5*N+pixel_shift_stack(1,idx_ps,idx_pt),1.5*M+1+pixel_shift_stack(2,idx_ps,idx_pt):2.5*M+pixel_shift_stack(2,idx_ps,idx_pt),idx_pt);
    
    Itemp = F(speckle_intensity_shift_crop(:,:,i).*I).*T_incoherent;
    I_image(:,:,i) = abs(iF(Itemp(N/2+1-Ncrop/2:N/2+1+Ncrop/2-1,M/2+1-Mcrop/2:M/2+1+Mcrop/2-1)));
    
    Ic_temp = F(abs(iF(F(speckle_field_shift_crop(:,:,i).*T_c).*Pupil_obj.*exp(-1j*pi*lambda*5*(fxx.^2+fyy.^2)))).^2);
    Ic_image(:,:,i) = abs(iF(Ic_temp(N/2+1-Ncrop/2:N/2+1+Ncrop/2-1,M/2+1-Mcrop/2:M/2+1+Mcrop/2-1)));
%     Ic_temp =F(speckle_field_shift_crop(:,:,i).*T_c).*Pupil_obj;
%     Ic_image(:,:,i) = iF(Ic_temp(N/2+1-Ncrop/2:N/2+1+Ncrop/2-1,M/2+1-Mcrop/2:M/2+1+Mcrop/2-1));

    if mod(i,100) == 0 || i == Nimg
        fprintf('Data generating process (%2d / %2d)\n',i,Nimg);
    end

end

%% Poisson + background noise

dark_current = 100;
photon_count = 5000;

% I_image = I_image/max(I_image(:))*photon_count;

I_image = imnoise(( I_image/max(I_image(:)).*photon_count + dark_current)*1e-12,'poisson')*1e12;
Ic_image = imnoise(( Ic_image/max(Ic_image(:)).*photon_count + dark_current)*1e-12,'poisson')*1e12;


%% Save file

savefile='res_speckle_shift';
save(savefile,'pscrop','lambda','NA_obj','I_image','I','speckle_intensity','pixel_shift_stack','speckle_intensity_shift_crop','Ic_image','speckle_field_shift_crop','-v7.3');

%%
% imwrite(uint16(I_image2(21:63,132:174,1)),'test.tiff');
% for j = 2:Nimg
% imwrite(uint16(I_image2(21:63,132:174,j)),'test.tiff','WriteMode', 'append');
% end