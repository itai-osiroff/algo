function y=frq_conv(x,fk,tt)

[n1,n2]=size(x);
[n1k,n2k]=size(fk);
if (n1k>n1)
    d1=(n1k-n1)/2;
    d2=(n2k-n2)/2;
    x=padarray(x,[d1,d2]);
    
end
y=fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(x))).*fk)));
if (n1k>n1)
    y=y(d1+1:end-d1,d2+1:end-d2);
end