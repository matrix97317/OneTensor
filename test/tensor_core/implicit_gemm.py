
import numpy as np
import torch
import  torch.nn.functional as F
# x: N,H,W,C
# w: OC,FH,FW,IC
# y: N,H,W,OC
N=1
H=7
W=7
IC=1
OC=1
FH=3
FW=3
OH=5
OW=5

stride_h=2
stride_w=2
pad_h=2
pad_w=2


gemm_m = N*OH*OW
gemm_n = OC
gemm_k = IC*FH*FW

np.random.seed(666)
x = np.random.rand(N,H,W,IC)
w = np.random.rand(OC,FH,FW,IC)
y = np.zeros((N,OH,OW,OC))
y = y.reshape((N*OH*OW,OC))

torch_x = torch.from_numpy(x.reshape(N,IC,H,W))
torch_w = torch.from_numpy(w.reshape(OC,IC,FH,FW))
out = F.conv2d(torch_x,torch_w,padding=2,stride=2)
print(out.shape)
print(out[0,0,:,:])


for gemm_i in range(gemm_m):
    for gemm_j in range(gemm_n):
        n = gemm_i//(OH*OW)
        npq_residual = gemm_i % (OH*OW)
        p = npq_residual // OW
        q = npq_residual % OW
        
        accum = 0
        for g_k in range(gemm_k):
            k = gemm_j
            c = g_k // (FH*FW)
            crs_residual = g_k % (FH*FW)
            r = crs_residual // FW
            s = crs_residual % FW
            
            ih = p*stride_h-pad_h+r
            iw = q*stride_w-pad_w+s            
            # print(f"N: {n}, H: {ih}, W: {iw}, C: {c}")
            # print(f"OC: {k}, FH: {r}, FW: {s}, C: {c}")
            if ih <0 or ih >= H or iw <0 or iw >= W:
                a=0
            else:
                a = x[n,ih,iw,c]
            
            b = w[k,r,s,c]
            accum+=a*b
        y[gemm_i,gemm_j]=accum
out = y.reshape((N,OH,OW,OC))
print(out[0,:,:,0])
            
            
            
            
            
    
            
            
            




