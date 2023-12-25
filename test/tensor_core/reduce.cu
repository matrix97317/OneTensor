#include <cub/cub.cuh>   
#include <one_tensor.h>

__global__ void block_reduce(float* data, float*out){
    typedef cub::BlockReduce<float, 128> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    float aggregate[1];
   
    aggregate[0] = BlockReduce(temp_storage).Sum(data[tid]);
    __syncthreads();
    if(threadIdx.x==0){
        // for(int i=0; i<8;i++){
        out[blockIdx.x]+=aggregate[0];
        // }
        
    }
}

int main(){
    OneTensor<float> data(Shape(1024));
    OneTensor<float> out(Shape(1,8));
    out.FillHostData(0);
    for(int i=0; i< data.shape.d0; i++){
        data.SetHostData(i,i);
    }

    data.sync_device();
    out.sync_device();
    dim3 grid(1024/128);
    dim3 block(128);
    block_reduce<<<grid,block>>>(data.deviceData<float>(),out.deviceData<float>());
    out.sync_device(false);
    out.HostDataView();


    return 0;
}