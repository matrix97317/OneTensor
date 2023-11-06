#include <one_tensor.h>

__global__ void load_store(float* in, float* out, size_t nums){
    size_t tid = blockIdx.x*blockDim.x+threadIdx.x;
    if (tid<nums){
        // Case 1,2
        out[tid] = in[tid];
        // Case 3
        // for(int i=0;i<nums; i++){
        //     float a = in[i];
        //     out[i] = a;
        // }
       
    }
}

int main(){
    
    Shape range(1000);

    OneTensor<float> bandwidth(range);
    OneTensor<float> measure_time(range);
    OneTensor<float> data_size(range);
    //range.d0+1
    int cnt=0;
    for(size_t i=1; i<range.d0+1;i++){
        std::cout<<cnt<<std::endl;
        cnt++;
        Shape probelm_size(1,i*5000);
        OneTensor<float> in_data(probelm_size);
        OneTensor<float> out_data(probelm_size);
        in_data.FillHostData(100);
        out_data.FillHostData(0);

        // Case 1: SMs * 1 Parallel
        // dim3 block(1,1,1);
        // dim3 grid(int(probelm_size.nums()/1)+1,1,1);
        // Case 2: SMs * 128 Parallel
        dim3 block(1024,1,1);
        dim3 grid(int(probelm_size.nums()/1024)+1,1,1);
        // Case 3: Single Thread
        // dim3 block(1,1,1);
        // dim3 grid(1,1,1);
        in_data.sync_device();
        out_data.sync_device();
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        GPU_Time((load_store<<<grid,block,0,stream>>>(in_data.deviceData<float>(),out_data.deviceData<float>(),probelm_size.nums())),
                stream,
                10,
                probelm_size.size<float>()*2);
        // Get measure time
        measure_time.SetHostData(mtime/(10) ,i-1);
        // Get data size
        data_size.SetHostData(i,i-1);
        // Get bandwidth
        float bw = 0.0;
        bw = (float((probelm_size.size<float>()*2))/(1024*1024*1024))/(mtime/(10*1000));
        bandwidth.SetHostData(bw,i-1);

        cudaStreamDestroy(stream);
    }
    bandwidth.SaveNpyFile<float>("bandwidth2.npy");
    measure_time.SaveNpyFile<float>("measure_time2.npy");
    data_size.SaveNpyFile<float>("datasize2.npy");
    return 0;
}