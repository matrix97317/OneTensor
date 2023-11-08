#include <cuda_runtime.h>
#include <iostream>
#include <one_tensor.h>
using namespace std;

__global__ void mma_m16n8k16(half  *a, 
                            half *b,
                            half *c,
                            half *d){
    uint32_t *ia0 =(uint32_t*)a;
    
    uint32_t *ib0 = (uint32_t*)b;
   
    uint32_t* c0 = (uint32_t*)c;
    uint32_t* d0 = (uint32_t*)d;
    

    int tid = blockDim.x*blockIdx.x+threadIdx.x; // wrap: 32Thread.
    int row = tid / 4;
    int col = tid % 4;

    // int ab_index = row*4+col;
    int a0_index = row*8+col;
    int a1_index = 64+row*8+col;
    int a2_index = row*8+col+4;
    int a3_index = 64+row*8+col+4;

    int b0_index = row*8+col;
    int b1_index = row*8+col+4;

    int cd0_index = row*4+col*1;
    int cd1_index = 32+row*4+col*1;

    asm volatile(
     "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
     "  {%0,%1},  "
     "  {%2,%3,%4,%5}, "
     "  {%6,%7}, "
     "  {%8,%9};  "
     :
     "=r"(d0[cd0_index]),"=r"(d0[cd1_index])
     :
     "r"(ia0[a0_index]),"r"(ia0[a1_index]),"r"(ia0[a2_index]),"r"(ia0[a3_index]),
     "r"(ib0[b0_index]),"r"(ib0[b1_index]),
     "r"(c0[cd0_index]),"r"(c0[cd1_index])
    );
}

int main(){
    OneTensor<half> a(Shape(16,16));
    OneTensor<half> b(Shape(8,16));
    OneTensor<half> c(Shape(16,8));
    OneTensor<half> d(Shape(16,8));
    c.FillHostData(half(0));
    d.FillHostData(half(0));
    for(int i=0; i<a.shape.d0; i++){
        for(int j=0; j<a.shape.d1; j++){
            int index = i*a.shape.d1+j;
            a.SetHostData(half(j),index);
            // b.SetHostData(j*1.0,index);
        }
    }
    for(int i=0; i<b.shape.d0; i++){
        for(int j=0; j<b.shape.d1; j++){
            int index = i*b.shape.d1+j;
            b.SetHostData(half(j),index);
            // b.SetHostData(j*1.0,index);
        }
    }
    a.HostDataView();
    b.HostDataView();
    // c.HostDataView();
    // d.HostDataView();

    a.sync_device();
    b.sync_device();
    c.sync_device();
    d.sync_device();
   
    dim3 gird(1,1,1);
    dim3 block(32,1,1);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    GPU_Time((mma_m16n8k16<<<gird,block,0,stream>>>(a.deviceData<half>(),
                                       b.deviceData<half>(),
                                       c.deviceData<half>(),
                                       d.deviceData<half>()
                                       )), stream,100,
                                       16*8*16);
    cudaStreamDestroy(stream);

    d.sync_device(false);
    d.HostDataView();
    int sum=0;
    for(int i=0;i<16;i++){
        sum+=i*i;
    }
    cout<<sum<<endl;

    
    
    return 0;
}