function [u_out_t_LF,u_out_t_hr,maskL,maskL_s, lf_ang_grid]=getLayered_To_LF_and_hr(sigt,x_max,x_stp,z_max,lambda,blr_w,scl_p,uin,max_used_x,zin,z_grid,lf_frq_res,smp_stp)

x_min=-x_max;


z_rng=z_max;
MFP=1/sigt;

x_grid=[x_min:x_stp:x_max];
x_u_grid=[-max_used_x:x_stp:max_used_x];
Nx=length(x_grid);
hNx=ceil(Nx/2);
Nxu=length(x_u_grid);
hNxu=ceil(Nxu/2);



hNxs=floor(max_used_x/(smp_stp*x_stp));
u_inds=hNx+[-hNxs:hNxs]*smp_stp;
[ug1,ug2]=ndgrid(u_inds);
u_inds=ug1(:)+(ug2(:)-1)*Nx; 
Ns=length(u_inds(:));

u_inds_hr1=hNx+[-hNxu+1:hNxu-1];
[ug1,ug2]=ndgrid(u_inds_hr1);
u_inds_hr=ug1(:)+(ug2(:)-1)*Nx; 
Ns_hr=length(u_inds_hr(:));



 

Nz=length(z_grid);



Ndiff=(Nx-Nxu)/2;

%kernels for propagation between layers

kb_all=gpuArray(getPropConvKerFreq3D_cyc(lambda,z_rng,x_max,x_stp));


for j=2:Nz
    z_stp=z_grid(j)-z_grid(j-1);
    kL(:,:,j)=gpuArray(getPropConvKerFreq3D_cyc(lambda,-z_stp,x_max,x_stp ));
end
uzin=unique(zin);
z_stp=z_grid(1);
for j=1:length(uzin)
  k0L(:,:,j)=gpuArray(getPropConvKerFreq3D_cyc(lambda,-z_stp-uzin(j),x_max,x_stp ));
end
if Nz>1
z_stp=z_grid(end)-z_grid(end-1);
end
Nl=size(uin,2);





%sample a  phase mask. Their angular distribution is controlled in a not
%so ideal way by the parameters blr_w, scl_p.
for j=1:Nz
    [mask_f,mask_s]=smpPhaseMask_GaussB(x_max,x_stp,blr_w,scl_p,sigt,z_stp,lambda);
            
    maskL(:,j)=gpuArray(mask_f(:));
    smp_inds_abr_grid=smpctr(Nx,smp_stp);  

    maskL_s(:,:,:,j)=mask_s(smp_inds_abr_grid,smp_inds_abr_grid,:);
end

v_stp=lambda/(2*x_max+x_stp);
v_max=lambda/(2*x_stp);
v_max=v_max-0.5*v_stp;

v_grid=[-v_max:v_stp:v_max];
[v1,v2]=ndgrid(v_grid);
r=(v1.^2+v2.^2);
mask=r<0.5^2;  %this is the numerical aperture of the objective



%grid of viewpoints in the aperture plane. directions are parametrized by sin(theta)   
lf_ang_grid=[-0.5:lf_frq_res:0.5];lf_ang_grid=lf_ang_grid-mean(lf_ang_grid);
Na=length(lf_ang_grid)-1;
Nas=Na^2;
vmaskL=zeros(Nx,Nx,Nas,'gpuArray');
vind=0;
for j2=1:length(lf_ang_grid)-1
    for j1=1:length(lf_ang_grid)-1
        b1d=lf_ang_grid(j1); b1u=lf_ang_grid(j1+1);
        b2d=lf_ang_grid(j2); b2u=lf_ang_grid(j2+1);
        %generate a mask of angles in the current bin
        vmask=((v1>b1d).*(v1<b1u).*(v2>b2d).*(v2<b2u));
        a=sum(sum(vmask.*mask))/sum(sum(vmask));
        if a<0.6
            vmask(:)=0;
        end
        vind=vind+1;
        vmaskL(:,:,vind)=vmask.*mask;
    end
end
u_out_t_LF=zeros(Ns,Nl,Nas,'gpuArray');

u_out_t_hr=zeros(Ns_hr,Nl,'gpuArray');


for j1=1:Nl
    

    e=reshape(uin(:,j1),Nxu,Nxu);
    e=padarray(e,Ndiff*[1,1]);
    %propagate once from the focus plae at the back of the volume to the
    %first enterence plane (closest to the source)


    tj=find(uzin==zin(j1));
    e=frq_conv(e,k0L(:,:,tj),'same');

   
    %loop through layers, multiply by the current phase mask and propagate
    %another step
    for j=1:Nz
             
             
       e(:)=e(:).*maskL(:,j);
       
       if j<Nz
       e=frq_conv(e,kL(:,:,j+1),'same');
       end
    end
    
    e=frq_conv(e,kb_all,'same');
    u_out_t_hr(:,j1)=e(u_inds_hr);
   
    for j=1:Nas
        ev=fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(e))).*vmaskL(:,:,j))));
        u_out_t_LF(:,j1,j)=ev(u_inds);
    end


  

    
end
lf_ang_grid = (lf_ang_grid(1:end-1) + lf_ang_grid(2:end)) * 0.5;
end
