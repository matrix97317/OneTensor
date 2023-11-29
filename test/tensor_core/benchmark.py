import numpy as np
import matplotlib.pyplot as plt

flops = np.load("./build/flops.npy").flatten()
cost_time = np.load("./build/cost_time.npy").flatten() / 1000

cublas_flops = np.load("./build/cublas_flops.npy").flatten()
cublas_cost_time = np.load("./build/cublas_cost_time.npy").flatten() / 1000


width=1
x = [i+width for i in range(len(flops))]
flops_s = ((flops/(1e9))/cost_time)
cublas_flops_s = ((cublas_flops/(1e9))/cublas_cost_time)

print(x)
plt.figure(1,figsize=(16,4),dpi=100)
plt.suptitle("TB32 K=16")
ax = plt.subplot(1,1,1)
ax.set_xlabel('Matrix Size (M=?, N=?, K=16)')
ax.set_ylabel('G-FLOPs/S ')
xticks = [str(int(v)) for v in flops]
print(xticks)

# ax.set_xticks(x)
ax.set_xticklabels(xticks,rotation=0)
ax.plot(x,flops_s)
ax.plot(x,cublas_flops_s)


plt.savefig("GEMM_TB32_benchmark.jpg")