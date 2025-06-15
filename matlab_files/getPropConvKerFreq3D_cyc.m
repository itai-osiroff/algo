function k=getPropConvKer(lambda,z_stp,x_max,x_stp )
bdr_fact=1;
wx_max=x_max*bdr_fact;
x_grid=[-wx_max:x_stp:wx_max];
[x_grid,y_grid]=ndgrid(x_grid);
v_stp=lambda/(2*x_max+x_stp);
v_max=lambda/(2*x_stp);
v_max=v_max-0.5*v_stp;

v_grid=[-v_max:v_stp:v_max];
[vx,vy]=ndgrid(v_grid);
r=(vx.^2+vy.^2);
mask=r<0.5^2;  %Can adjust this threshold 
p=sqrt(1-r.*mask);
k=exp(-2*pi*i/lambda*(p.*mask)*z_stp).*mask;

return
