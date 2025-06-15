function smp_inds=smpctr(N,smp_stp,sN);
hN=ceil(N/2);
hsN=floor(hN/smp_stp);
if exist('sN','var')
if (floor(sN/2)<hsN)
    hsN=floor(sN/2);
end
end
smp_inds=hN+[-hsN:hsN]*smp_stp;
   