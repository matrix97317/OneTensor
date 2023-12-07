#include <cuda_runtime.h>
#include <iostream>
#include <one_tensor.h>
#include "gemm_api_tb256.cuh"
// #include "gemm_api_tb128.cuh"
using namespace std;

#define MEASURE_TIME 1
#define USE_CUBLAS 1
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
        OneTensor<half> b(Shape(K, N));
        OneTensor<float> c(Shape(M, N));
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
                b.SetHostData(half(1), index);
            }
        }

        // a.HostDataView();
        // b.HostDataView();

        a.sync_device();
        b.sync_device();
        c.sync_device();

        dim3 gird(16, div_ceil(M, (WARP_ROW_NUM * MMA_ROW_NUM * MMA_M)), div_ceil(N, (WARP_COL_NUM * MMA_COL_NUM * MMA_N * 16)));
        dim3 block(32 * 8, 1, 1);
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        if (!(USE_CUBLAS))
        {
            if (!(MEASURE_TIME))
            {
                gemm_fp16fp32_tb256<<<gird, block, 0, stream>>>(a.deviceData<half>(),
                                                                b.deviceData<half>(),
                                                                c.deviceData<float>(),
                                                                M, N, K);
            }
            else
            {
                printf("our: cnt: %d\n", cnt);
                GPU_Time((gemm_fp16fp32_tb256<<<gird, block, 0, stream>>>(a.deviceData<half>(), b.deviceData<half>(), c.deviceData<float>(), M, N, K)),
                         stream,
                         200,
                         0);
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
                    blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                    &alpha, b.deviceData<half>(), CUDA_R_16F, N, a.deviceData<half>(), CUDA_R_16F, K, &beta, c.deviceData<float>(), CUDA_R_32F, N,
                    CUBLAS_COMPUTE_32F, algo_list[1]));
            }
            else
            {
                printf("cublas: cnt: %d\n", cnt);
                GPU_Time((cublasGemmEx(
                             blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                             &alpha, b.deviceData<half>(), CUDA_R_16F, N, a.deviceData<half>(), CUDA_R_16F, K, &beta, c.deviceData<float>(), CUDA_R_32F, N,
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