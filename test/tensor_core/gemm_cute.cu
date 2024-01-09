#include <cuda_runtime.h>
#include <iostream>
#include <one_tensor.h>
#include "gemm_api_cute.cuh"

using namespace std;

#define MEASURE_TIME 1
#define USE_CUBLAS 0
#define DEBUG 0
#define EVAL_TIMES 30
int main()
{

    int measure_times = EVAL_TIMES;
    OneTensor<double> flops(Shape(EVAL_TIMES));
    OneTensor<double> cost_time(Shape(EVAL_TIMES));
    for (int cnt = 1; cnt < measure_times; cnt++)
    {
        float excute_time=0.0;
        
        int M = 16 * 4 * 4 * (cnt + 1);
        int N = 16 * 4 * 4 * (cnt + 1);
        int K=16*4*4*(cnt+1);
        // int K = 16 * 4 * 2 * (1 + 1);
        printf("M: %d\n", M);

        OneTensor<half> a(Shape(M, K));
        OneTensor<half> b(Shape(N, K));
        OneTensor<half> c(Shape(M, N));
        c.FillHostData(0);
        for (int i = 0; i < a.shape.d0; i++)
        {
            for (int j = 0; j < a.shape.d1; j++)
            {
                int index = i * a.shape.d1 + j;
                a.SetHostData(half(1), index);
            }
        }
        for (int i = 0; i < b.shape.d0; i++)
        {
            for (int j = 0; j < b.shape.d1; j++)
            {
                int index = i * b.shape.d1 + j;
                b.SetHostData(half(i), index);
            }
        }

        // a.HostDataView();
        // b.HostDataView();

        a.sync_device();
        b.sync_device();
        c.sync_device();
        ConfigAttr::GemmConfig<half,128,128> config;

        dim3 gird(div_ceil(N, config.TB_N), div_ceil(M,config.TB_M));
        dim3 block(config.block_thread_size);
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        if (!(USE_CUBLAS))
        {
            if (!(MEASURE_TIME))
            {
                int smem_size = config.smem_size;
                printf("smem size: %d\n",smem_size);
                cudaFuncSetAttribute(cute_gemm_kernel<half,half,decltype(config)>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem_size);
                cute_gemm_kernel<half,half,decltype(config)><<<gird, block, smem_size,stream>>>(
                                                                a.deviceData<half>(),
                                                                b.deviceData<half>(),
                                                                c.deviceData<half>(),
                                                                M, N, K);
            }
            else
            {
                printf("our: cnt: %d\n", cnt);
                int smem_size = config.smem_size;
                cudaFuncSetAttribute(cute_gemm_kernel<half,half,decltype(config)>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem_size);
                GPU_Time((cute_gemm_kernel<half,half,decltype(config)><<<gird, block, smem_size,stream>>>(
                                                                a.deviceData<half>(),
                                                                b.deviceData<half>(),
                                                                c.deviceData<half>(),
                                                                M, N, K)),stream, 200, 0);
                excute_time=mtime;
            }
        }
        else
        {

            float alpha = 1.0;
            float beta = 0.0;
            cudaDeviceSynchronize();
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
            if (!(MEASURE_TIME))
            {
                (cublasGemmEx(
                    blas_handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                    &alpha, b.deviceData<half>(), CUDA_R_16F, K, a.deviceData<half>(), CUDA_R_16F, K, &beta, c.deviceData<half>(), CUDA_R_16F, N,
                    CUBLAS_COMPUTE_32F, algo_list[1]));
            }
            else
            {
                printf("cublas: cnt: %d\n", cnt);
                GPU_Time((cublasGemmEx(
                             blas_handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                             &alpha, b.deviceData<half>(), CUDA_R_16F, K, a.deviceData<half>(), CUDA_R_16F, K, &beta, c.deviceData<half>(), CUDA_R_16F, N,
                             CUBLAS_COMPUTE_32F, algo_list[1])),
                         stream, 200, 0);
                excute_time=mtime;

            }
            cublasDestroy(blas_handle);
        }
        if(DEBUG){
            c.sync_device(false);
            c.HostDataView();
        }

        if (MEASURE_TIME)
        {
            cost_time.SetHostData((excute_time / (200)), cnt - 1);
            flops.SetHostData(M, cnt - 1);
        }
        cudaStreamDestroy(stream);
    }

    if (MEASURE_TIME)
    {
        if (USE_CUBLAS)
        {   
            printf("save cublas cost time.\n");
            cost_time.SaveNpyFile<double>("cublas_cost_time.npy");
            flops.SaveNpyFile<double>("cublas_flops.npy");
        }
        else
        {
            printf("save our cost time.\n");
            cost_time.SaveNpyFile<double>("v0_cost_time.npy");
            flops.SaveNpyFile<double>("v0_flops.npy");
        }
    }

    return 0;
}