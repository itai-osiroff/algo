function iLF=interp2D_frw(LF,i1,i2)
%LF=padarray(LF,[1,1],0,'both');
%i1+i1+1; i2=i2+1; 
[n1,n2]=size(LF);
iLF=zeros(n1,n2,'gpuArray');
%map=(i1>=2).*(i1<=(n1-1)).*(i2>=2).*(i2<=(n2-1))
map=(i1>1.001).*(i1<(n1-0.001)).*(i2>1.000001).*(i2<(n2-0.00001));

i1=min(max(i1,1),n1);
i2=min(max(i2,1),n2);


fi1=floor(i1);
fi2=floor(i2);



w1(:,:,1)=fi1+1-i1; w1(:,:,2)=1-w1(:,:,1);
w2(:,:,1)=fi2+1-i2; w2(:,:,2)=1-w2(:,:,1);

for j3=0:1
    for j4=0:1
        mask=zeros(n1,n2,'gpuArray'); mask(1+j3:2:end,1+j4:2:end)=1;
        %valid_inds=find(map(1+j3:2:end,1+j4:2:end));
        valid_inds=find(map.*mask);
        
        for j1=0:1
            for j2=0:1

                %inds=(fi1(1+j3:2:end,1+j4:2:end)+j1)+(fi2(1+j3:2:end,1+j4:2:end)+j2-1)*n1;

                inds=fi1(valid_inds)+j1+(fi2(valid_inds)+j2-1)*n1;

                targ_v=LF(valid_inds).*w1(valid_inds+(j1)*n1*n2).*w2(valid_inds+(j2)*n1*n2);
                %LF(2+j3:2:end-1,2+j4:2:end-1).*w1(1+j3:2:end,1+j4:2:end,j1+1).*w2(1+j3:2:end,1+j4:2:end,j2+1);
                
                %size(LF(2+j3:2:end-1,2+j4:2:end-1))
                %size(w1(1+j3:2:end,1+j4:2:end,j1+1))
                %iLF(inds)=iLF(inds)+LF(2+j3:2:end-1,2+j4:2:end-1).*w1(1+j3:2:end,1+j4:2:end,j1+1).*w2(1+j3:2:end,1+j4:2:end,j2+1);
                iLF(inds)=iLF(inds)+targ_v;
            end
        end
    end
end
%iLF=iLF(2:end-1,2:end-1);

