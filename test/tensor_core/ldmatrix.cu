#include <cuda_runtime.h>
#include <iostream>
#include <one_tensor.h>
using namespace std;

__global__ void ldmatrix(half  *a, 
                             half  *b){

    int tid = blockDim.x*blockIdx.x+threadIdx.x; // wrap: 32Thread.
    uint32_t *ia0 = (uint32_t*)a;
    uint32_t *ib0 = (uint32_t*)b;
    // a (16,16) b(16,16)
    // ia0 (16,8) ib0(16,8)
    // T0-T15 , T16-T31
    __shared__ uint32_t smem[16*8];
    int idx = tid*4;
    smem[idx+0]=ia0[idx+0];
    smem[idx+1]=ia0[idx+1];
    smem[idx+2]=ia0[idx+2];
    smem[idx+3]=ia0[idx+3];
    __syncthreads();
    int row = tid % 16;
    int col = tid / 16;
    unsigned smem_ptr = static_cast<unsigned>(__cvta_generic_to_shared(&smem[row*8+col*4]));

    uint32_t A[4];

    asm volatile(
     "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
     "  {%0,%1,%2,%3},  "
     "  [%4];  "
     :
     "=r"(A[0]),"=r"(A[1]),"=r"(A[2]),"=r"(A[3])
     :
     "r"(smem_ptr)
    );
    __syncthreads();
    int iRow = tid / 4;
    int iCol = tid % 4;

    int b0 = iRow*8+iCol;
    int b1 = 64+iRow*8+iCol; 
    int b2 = iRow*8+iCol+4;
    int b3 = 64+iRow*8+iCol+4;
    ib0[b0] = A[0];
    ib0[b1] = A[1];
    ib0[b2] = A[2];
    ib0[b3] = A[3];
}

int main(){
    OneTensor<half> a(Shape(16,16));
    OneTensor<half> b(Shape(16,16));
    b.FillHostData(0);
    for(int i=0; i<a.shape.d0; i++){
        for(int j=0; j<a.shape.d1; j++){
            int index = i*a.shape.d1+j;
            a.SetHostData(j*1,index);
            // b.SetHostData(j*1.0,index);
        }
    }
    // a.HostDataView();
    // b.HostDataView();
 
    a.sync_device();
    b.sync_device();
    
   
    dim3 gird(1,1,1);
    dim3 block(32,1,1);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    ldmatrix<<<gird,block,0,stream>>>(a.deviceData<half>(),
                                       b.deviceData<half>()
                                       );
    // GPU_Time((ldmatrix<<<gird,block,0,stream>>>(a.deviceData<half>(),b.deviceData<half>())), 
    //             stream,
    //             10,
    //             Shape(16,16).size<half>()*2);
    cudaStreamDestroy(stream);

    // b.sync_device(false);
    // b.HostDataView(); 
    return 0;
}