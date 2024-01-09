#include<one_tensor.h>
#include "img2col_sparse_conv.cuh"

#define BS 1
#define H 7
#define W 7
#define IC 2048
#define KH 3
#define KW 3
#define OC 2048
#define PAD_H 2
#define PAD_W 2
#define S_H 2
#define S_W 2


using DT = half;


int main(){
    OneTensor<DT> inputs_data(Shape(BS,H,W,IC));
    OneTensor<DT> weight_data(Shape(OC,KH,KW,IC));

    // OneTensor<DT> inputs_data("x.npy");
    // OneTensor<DT> weight_data("w.npy");
    // inputs_data.HostDataView();
    // weight_data.HostDataView();

    int oh=0;
    int ow=0;
    get_oh_ow(&oh,&ow,H,W,KH,KW,PAD_H,PAD_W,S_H,S_W);
    printf("oh: %d, ow: %d\n",oh,ow);

    OneTensor<DT> outputs_data(Shape(BS,oh,ow,OC));
    outputs_data.FillHostData(0);

    OneTensor<DT> i2c_inputs_data(Shape(BS*oh*ow,KH*KW*IC));
    OneTensor<int> offset_vec(Shape(BS*oh*ow,KH*KW*IC));
    i2c_inputs_data.FillHostData(0);
    offset_vec.FillHostData(0);

    pre_comp(offset_vec.hostData<int>(),BS,oh,ow,KH,KW,IC,H,W,PAD_H,PAD_W,S_H,S_W);

    int M=BS*oh*ow;
    int K=KH*KW*IC;
    int N=OC;

    // for(int bs=0; bs<inputs_data.shape.d0;bs++){
    //     for(int h=0; h<inputs_data.shape.d1; h++){
    //         for(int w=0; w<inputs_data.shape.d2; w++){
    //             for(int c=0; c<inputs_data.shape.d3; c++){
    //                 int index = bs*inputs_data.shape.strides(0)+\
    //                             h*inputs_data.shape.strides(1)+\
    //                             w*inputs_data.shape.strides(2)+c;
    //                 inputs_data.SetHostData(index,index);
    //             }
    //         }
    //     }
    // }
    offset_vec.sync_device();
    inputs_data.sync_device();
    i2c_inputs_data.sync_device();
    weight_data.sync_device();
    outputs_data.sync_device();

    cudaBindTexture(NULL,texref,offset_vec.deviceData<int>(),offset_vec.shape.size<int>());
    
    dim3 grid(div_ceil(BS*oh*ow*KH*KW*IC/1, 256));
    dim3 block(256);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // step0. img2col
    GPU_Time((img2col<DT><<<grid,block,0,stream>>>(
                                    inputs_data.deviceData<DT>(),
                                    i2c_inputs_data.deviceData<DT>(),
                                    BS*oh*ow*KH*KW*IC/1,1)),stream,100,0);
    // i2c_inputs_data.sync_device(false);
    // i2c_inputs_data.HostDataView();
    // step1. MatMul
    float alpha=1.0;
    float beta=0.0;
    cublasHandle_t blas_handle;
    cublasCreate(&blas_handle);
    cublasSetStream(blas_handle, stream);
    cublasGemmAlgo_t algo_list[19] = {
        CUBLAS_GEMM_DFALT,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        CUBLAS_GEMM_DFALT_TENSOR_OP,
        CUBLAS_GEMM_ALGO0_TENSOR_OP,
        CUBLAS_GEMM_ALGO1_TENSOR_OP,
        CUBLAS_GEMM_ALGO2_TENSOR_OP,
        CUBLAS_GEMM_ALGO3_TENSOR_OP,
        CUBLAS_GEMM_ALGO4_TENSOR_OP,
        CUBLAS_GEMM_ALGO5_TENSOR_OP,
        CUBLAS_GEMM_ALGO6_TENSOR_OP,
        CUBLAS_GEMM_ALGO7_TENSOR_OP,
        CUBLAS_GEMM_ALGO8_TENSOR_OP,
        CUBLAS_GEMM_ALGO9_TENSOR_OP,
        CUBLAS_GEMM_ALGO10_TENSOR_OP,
        CUBLAS_GEMM_ALGO11_TENSOR_OP,
        CUBLAS_GEMM_ALGO12_TENSOR_OP,
        CUBLAS_GEMM_ALGO13_TENSOR_OP,
        CUBLAS_GEMM_ALGO14_TENSOR_OP,
        CUBLAS_GEMM_ALGO15_TENSOR_OP};
    GPU_Time(cublasGemmEx(
                    blas_handle, 
                    CUBLAS_OP_T, 
                    CUBLAS_OP_N, 
                    N, M, K,
                    &alpha, weight_data.deviceData<DT>(), CUDA_R_16F, K, 
                    i2c_inputs_data.deviceData<DT>(), CUDA_R_16F, K, 
                    &beta, outputs_data.deviceData<DT>(), CUDA_R_16F, N,
                    CUBLAS_COMPUTE_32F, algo_list[1]),stream,100,0);
    // step2. Epilogue
    // outputs_data.sync_device(false);
    // outputs_data.HostDataView();
    get_last_cuda_err();
    cudaUnbindTexture(texref);
    cudaStreamDestroy(stream);

    return 0;
}