function [mask_f,mask_s]=smpPhaseMask(x_max,x_stp,blr_w,scl_p,sigt,z_stp,lambda)
dim=3;
bdr_fact=2;
wx_max=x_max*bdr_fact;
wx_grid=[-wx_max:x_stp:wx_max];
x_grid=[-x_max:x_stp:x_max];
ii=find((abs(wx_grid)<=(x_max+x_stp/10)));


[x_grid,y_grid]=ndgrid(x_grid,x_grid);
[wx_grid,wy_grid]=ndgrid(wx_grid,wx_grid);



v_max=lambda/x_stp/2;
v_stp=lambda/x_max/2;
v_max=ceil(v_max/v_stp)*v_stp;

v_grid=[-v_max:v_stp:v_max];
[vx_grid,vy_grid]=ndgrid(v_grid,v_grid);

Nv=length(v_grid);
hNv=floor(Nv/2);

r_grid=sqrt(vx_grid.^2+vy_grid.^2);
mask=r_grid<0.9;
mask=r_grid<0.5;



blr_f1=exp(-0.5*[-blr_w*5:x_stp:blr_w*5].^2/blr_w.^2);
blr_f2=exp(-0.5*[-blr_w*5:x_stp:blr_w*5].^2/(2*blr_w).^2);

fl=length(blr_f1);

a_f=randn(Nv+2*fl-2,Nv+2*fl-2);
a_f=conv2(blr_f2,blr_f2',a_f,'valid');
a_f=round(a_f/mean(abs(a_f(:)))*1);
a_f=conv2(blr_f1,blr_f1',a_f,'valid');
df=conv2(a_f,[-1 1],'valid');
mean_s=sqrt(mean(df(:).^2))/x_stp;
a_f=a_f/mean_s*scl_p;

%a_f=y_grid*0.08/lambda;
%a_f=y_grid.^2/x_max*0.05/lambda;


mask_f0=exp(-2*pi*i*a_f);
%mask_f0(:)=1;
mu=exp(-sigt/2*z_stp); 
mask_f=mu+mask_f0*(1-exp(-sigt*z_stp)); 

mask_s(:,:,1)=conv2(a_f,[-1;0;1],'same')/(2*x_stp)*lambda;
mask_s(:,:,2)=conv2(a_f,[-1,0,1],'same')/(2*x_stp)*lambda;
mask_s = gpuArray(mask_s);
return
