# OneTensor
This is a simple and easy-to-use Tensor Library.

# 1. Installation
``` bash
$ cd one_tensor
$ mkdir build
$ cd build
$ cmake ..
$ make
$ make install
```

# 2. Usage
``` C++
#include <one_tensor.h>

OneTensor<int8_t> a(Shape(8,16));
OneTensor<int8_t> b(Shape(8,16));
OneTensor<int32_t> c(Shape(8,8));
OneTensor<int32_t> d(Shape(8,8));
c.FillHostData(0);
d.FillHostData(0);
for(int i=0; i<a.shape.d0; i++){
    for(int j=0; j<a.shape.d1; j++){
        int index = i*a.shape.d1+j;
        a.SetHostData(j,index);
        b.SetHostData(j,index);
    }
}
a.HostDataView();

a.sync_device();
b.sync_device();
c.sync_device();
d.sync_device(); // Send Host Data to Device
d.sync_device(false); // Send Device Data to Host

int8_t *  ptr = a.deviceData<int8_t>();

// Measure your kernel time.
cudaStream_t stream;
cudaStreamCreate(&stream);
int iter_nums = 100;
GPU_Time((your_func)), stream, iter_nums);

// Load .npy or npz file from numpy
OneTensor<float> npz_file("./test_data.npz","data");
std::cout<<npz_file.shape<<std::endl;
npz_file.HostDataView();

OneTensor<float> npy_file("./test_data.npy");
std::cout<<npy_file.shape<<std::endl;
npy_file.HostDataView();

```