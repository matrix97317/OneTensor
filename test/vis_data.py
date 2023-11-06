import numpy as np
import matplotlib.pyplot as plt

mtime3 = np.load("measure_time3.npy").flatten()
bw3 = np.load("bandwidth3.npy").flatten()
datasize3 = np.load("datasize3.npy").flatten()*8*5000

mtime2 = np.load("measure_time2.npy").flatten()
bw2 = np.load("bandwidth2.npy").flatten()
datasize2 = np.load("datasize2.npy").flatten()*8*5000

mtime1 = np.load("measure_time1.npy").flatten()
bw1 = np.load("bandwidth1.npy").flatten()
datasize1 = np.load("datasize1.npy").flatten()*8*5000

plt.figure(1,figsize=(16,4),dpi=100)
plt.suptitle("RTX 3080TI SMs=68, Core/SM=128,All Cores=8704")
ax = plt.subplot(1,4,1)
ax.set_xlabel('Data Size (bytes)(R+W)')
ax.set_ylabel('Latencty (ms)')
ax.plot(datasize3,mtime3)
ax.plot(datasize2,mtime2)
ax.plot(datasize1,mtime1)


ax = plt.subplot(1,4,2)
ax.set_title(" Grid=1,Block=1, P=1, ")
ax.set_xlabel('Data Size (bytes)(R+W)')
ax.set_ylabel('Bandwidth (GB/s)')
ax.plot(datasize3,bw3)

ax = plt.subplot(1,4,3)
ax.set_title(" Grid=N,Block=1, P=68, ")
ax.set_xlabel('Data Size (bytes)(R+W)')
ax.set_ylabel('Bandwidth (GB/s)')
ax.plot(datasize1,bw1)

ax = plt.subplot(1,4,4)
ax.set_title(" Grid=N/128,Block=128, P=8704, ")
ax.set_xlabel('Data Size (bytes)(R+W)')
ax.set_ylabel('Bandwidth (GB/s)')
ax.plot(datasize2,bw2)

plt.savefig("TestGPU_Bandwidth.jpg")

