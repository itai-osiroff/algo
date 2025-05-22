
syms x a c f1 f2 real

L1=Lmat(f1);
L2=Lmat(f2);
T1=Tmat(f1);
T2=Tmat(f2);

T=T2*T1*L1*T1;
q0=T1*L1*T1*[x;a]
q1=T*[x;a];
q=q1;
q(1)=q(1)-c;
q=T2*L2*q;
simplify(q)



f1=1; 
x_rng=0.03%*20;
v_rng=0.5%/20;
xv=[-2:1:2]/2*(x_rng);
av=[-0.5:0.05:0.5]/0.5*v_rng;
%av=[-0.4:0.2:0.4]/0.5*v_rng;
apt_size=2*v_rng*f1 %%aperture size after passing through f1 is defined by the max angle we image

c_stp=apt_size/5;   % want 5 viewpoints
c_min=4;
c_list=[-c_min:c_min]*c_stp;

%to set f2: original image is [-x_rng, x_rng]. After imaging through f1 f2
%demagnified to [-x_rng, x_rng]*f2/f1. To avoid cross talks between viewpoint images, this needs to be smaller than
%lenslet aperture [-c_stp/2 c_stp/2] so
f2=c_stp/2*f1/x_rng*0.9

L1=Lmat(f1);
L2=Lmat(f2);
T1=Tmat(f1);
T2=Tmat(f2);
figure, hold on;
cols='rgbcm'
for j1=1:length(xv)
    for j2=1:length(av)
        q=[xv(j1);av(j2)];
        z=0;
        qL=[[q(1);z]];
        q=T1*q;
        z=z+f1;
        qL=[qL,[q(1);z]];
        q=T2*T1*L1*q;
         z=z+f1+f2;
        qL=[qL,[q(1);z]];

        ci=round(q(1)/c_stp+c_min+1);
        c=c_list(ci);
        tq=q; tq(1)=tq(1)-c;
        tq=T2*L2*tq;
        q=tq; q(1)=q(1)+c;
        z=z+f2;
        qL=[qL,[q(1);z]];
        %plot(qL(1,:),qL(2,:),'.-','color',cols(j2))
        plot(qL(1,:),qL(2,:),'.-','color',cols(j1))
        
    end
end
