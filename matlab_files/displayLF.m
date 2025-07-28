function LFf=displayLF(LF,bdr)
LF=padarray(LF,[bdr,bdr],0.5*max(LF(:)),'both');
[n1,n2,n3,n4]=size(LF);

LFf=zeros(n1*n3,n2*n4);

for j1=1:n3
    for j2=1:n4
        LFf(n1*(j1-1)+1:n1*j1,n2*(j2-1)+1:n2*j2)=LF(:,:,j1,j2);
    end
end
figure, imagesc(LFf)