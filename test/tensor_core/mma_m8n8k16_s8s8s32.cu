#include <cuda_runtime.h>
#include <iostream>
#include <one_tensor.h>
using namespace std;

__global__ void mma_m8n8k16(int8_t  *a, 
                            int8_t  *b,
                            int32_t *c,
                            int32_t *d){
    uint32_t * ia = (uint32_t*)a;
    uint32_t * ib = (uint32_t*)b;
    int tid = blockDim.x*blockIdx.x+threadIdx.x; // wrap: 32Thread.
    int row = tid / 4;
    int col = tid % 4;

    int ab_index = row*4+col;
    int cd_index = row*8+col*2;
    asm volatile(
     "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
     "  {%0,%1},  "
     "  {%2}, "
     "  {%3}, "
     "  {%4,%5};  "
     :
     "=r"(d[cd_index]),"=r"(d[cd_index+1])
     :
     "r"(ia[ab_index]),
     "r"(ib[ab_index]),
     "r"(c[cd_index]),"r"(c[cd_index+1])
    );
}

int main(){
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
    // a.HostDataView();
    // b.HostDataView();
    // c.HostDataView();
    d.HostDataView();

    a.sync_device();
    b.sync_device();
    c.sync_device();
    d.sync_device();
   
    dim3 gird(1,1,1);
    dim3 block(32,1,1);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    GPU_Time(
    (mma_m8n8k16<<<gird,block,0,stream>>>(a.deviceData<int8_t>(),
                                       b.deviceData<int8_t>(),
                                       c.deviceData<int32_t>(),
                                       d.deviceData<int32_t>()
                                       )), stream,100,
                                       8*16*8);
    cudaStreamDestroy(stream);

    d.sync_device(false);
    d.HostDataView();
    // int sum=0;
    // for(int i=0;i<16;i++){
    //     sum+=i*i;
    // }
    // cout<<sum<<endl;

    
    
    return 0;
}