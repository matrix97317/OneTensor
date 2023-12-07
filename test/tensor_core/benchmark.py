import numpy as np
import matplotlib.pyplot as plt

v0flops = np.load("./build/v0_flops.npy").flatten()[:-1]
v0cost_time = np.load("./build/v0_cost_time.npy").flatten()[:-1] / 1000


# v1flops = np.load("./build/v1_flops.npy").flatten()
# v1cost_time = np.load("./build/v1_cost_time.npy").flatten() / 1000

cublas_flops = np.load("./build/cublas_flops.npy").flatten()[:-1]
cublas_cost_time = np.load("./build/cublas_cost_time.npy").flatten()[:-1] / 1000

K=v0flops
TB=256
width=1
x = [i for i in range(len(v0flops))]
v0flops_s = ((v0flops*K*v0flops/(1e9))/v0cost_time)
# v1flops_s = ((v1flops/(1e9))/v1cost_time)

cublas_flops_s = ((cublas_flops*K*cublas_flops/(1e9))/cublas_cost_time)

print(x)
plt.figure(1,figsize=(16,8),dpi=100)
plt.suptitle(f"FP16FP16-FP32-TB{TB}")
ax = plt.subplot(1,1,1)
ax.set_xlabel('Matrix Size (M=N=K)')
ax.set_ylabel('G-FLOPs/S ')
ax.set_ylim(0,60000)
xticks = [str(int(v)) for v in cublas_flops]
print(xticks)

ax.set_xticks(x)
ax.set_xticklabels(xticks,rotation=90)
ax.plot(x,v0flops_s,label='Our')
# ax.plot(x,v1flops_s)
ax.plot(x,cublas_flops_s,label='cublas')
ax.legend()

plt.savefig(f"GEMM_TB{TB}_MNK_B_benchmark.jpg")