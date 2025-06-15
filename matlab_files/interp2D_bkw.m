function iLF=interp2D_bkw(LF,i1,i2)
LF=padarray(LF,[1,1],0,'both');
i1=i1+1; i2=i2+1; 
[n1,n2,n3]=size(LF);
iLF=zeros(n1,n2,n3,'gpuArray');

map=(i1>=1).*(i1<=(n1-1)).*(i2>=1).*(i2<=(n2-1));
i1=min(max(i1,1),n1-1);
i2=min(max(i2,1),n2-1);


fi1=floor(i1);
fi2=floor(i2);



w1(:,:,1)=fi1+1-i1; w1(:,:,2)=1-w1(:,:,1);
w2(:,:,1)=fi2+1-i2; w2(:,:,2)=1-w2(:,:,1);


for j1=0:1
    for j2=0:1

                inds0=(fi1+j1)+(fi2+j2-1)*n1;
                for k=1:n3
                    inds=inds0+(k-1)*n1*n3;
                   iLF(2:end-1,2:end-1,k)=iLF(2:end-1,2:end-1,k)+LF(inds).*w1(:,:,j1+1).*w2(:,:,j2+1);
                end
        end

end
iLF=iLF(2:end-1,2:end-1,:);

