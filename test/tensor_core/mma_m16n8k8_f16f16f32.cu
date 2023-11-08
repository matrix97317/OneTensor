#include <cuda_runtime.h>
#include <iostream>
#include <one_tensor.h>
using namespace std;

__global__ void mma_m16n8k8(half  *a, 
                            half  *b,
                            float *c,
                            float *d){
    uint32_t *ia0 =(uint32_t*) a;
    uint32_t *ia1 = (uint32_t*)(&a[8*8]);
    uint32_t *ib = (uint32_t*)b;
    float* c0 = c;
    float* c1 =(float*)(&c[8*8]);
    float* d0 = d;
    float* d1 = (float*)(&d[8*8]);

    int tid = blockDim.x*blockIdx.x+threadIdx.x; // wrap: 32Thread.
    int row = tid / 4;
    int col = tid % 4;

    int ab_index = row*4+col;
    int cd_index = row*8+col*2;
    asm volatile(
     "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
     "  {%0,%1,%2,%3},  "
     "  {%4,%5}, "
     "  {%6}, "
     "  {%7,%8,%9,%10};  "
     :
     "=f"(d0[cd_index]),"=f"(d0[cd_index+1]),"=f"(d1[cd_index]),"=f"(d1[cd_index+1])
     :
     "r"(ia0[ab_index]),"r"(ia1[ab_index]),
     "r"(ib[ab_index]),
     "f"(c0[cd_index]),"f"(c0[cd_index+1]),"f"(c1[cd_index]),"f"(c1[cd_index+1])
    );
}

int main(){
    OneTensor<half> a(Shape(16,8));
    OneTensor<half> b(Shape(8,8));
    OneTensor<float> c(Shape(16,8));
    OneTensor<float> d(Shape(16,8));
    c.FillHostData(0);
    d.FillHostData(0);
    for(int i=0; i<a.shape.d0; i++){
        for(int j=0; j<a.shape.d1; j++){
            int index = i*a.shape.d1+j;
            a.SetHostData(j*0.1,index);
            // b.SetHostData(j*1.0,index);
        }
    }
    for(int i=0; i<b.shape.d0; i++){
        for(int j=0; j<b.shape.d1; j++){
            int index = i*b.shape.d1+j;
            b.SetHostData(j*0.1,index);
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
    GPU_Time((mma_m16n8k8<<<gird,block,0,stream>>>(a.deviceData<half>(),
                                       b.deviceData<half>(),
                                       c.deviceData<float>(),
                                       d.deviceData<float>()
                                       )), stream,100, 16*8*8);
    cudaStreamDestroy(stream);

    d.sync_device(false);
    d.HostDataView();
    int sum=0;
    for(int i=0;i<8;i++){
        sum+=i*i;
    }
    cout<<sum<<endl;

    
    
    return 0;
}