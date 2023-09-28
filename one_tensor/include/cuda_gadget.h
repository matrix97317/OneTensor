#ifndef __CUDA_GADGET_H__
#define __CUDA_GADGET_H__
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>
using namespace std::chrono;

// CUDA Runtime API: Capture Error
#define cuda_checker(EXPER)                                   \
    {                                                         \
        cudaError_t err = (EXPER);                            \
        if (err != cudaSuccess) {                             \
            const char *err_name = cudaGetErrorName(err);     \
            const char *err_string = cudaGetErrorString(err); \
            printf("FILE: %s , LINE: %d , Error [%s] : %s\n", \
                   __FILE__, __LINE__, err_name, err_string); \
        }                                                     \
    }

#define get_last_cuda_err() \
    cuda_checker(cudaGetLastError())

// CUBLAS : Capture Error
const char *getCublasErrorName(cublasStatus_t err);

#define cublas_checker(EXPER)                               \
    {                                                       \
        cublasStatus_t err = (EXPER);                       \
        if (err != CUBLAS_STATUS_SUCCESS) {                 \
            const char *err_name = getCublasErrorName(err); \
            printf("FILE: %s , LINE: %d , Error [%s]\n",    \
                   __FILE__, __LINE__, err_name);           \
        }                                                   \
    }

// CUDA Measure Cost Time
#define GPU_Time(EXPER, cu_stream,iter)           \
    {                                             \
        for(int i=0; i<(iter+100); i++){              \
            (EXPER);                              \
        }                                         \
        cudaEvent_t start, stop;                  \
        float time = 0.0;                         \
        cudaEventCreate(&start);                  \
        cudaEventCreate(&stop);                   \
        cudaEventRecord(start, cu_stream);        \
        for(int i=0; i<(iter); i++){              \
            (EXPER);                              \
        }                                         \
        cudaEventRecord(stop, cu_stream);         \
        cudaEventSynchronize(stop);               \
        cudaEventElapsedTime(&time, start, stop); \
        cudaEventDestroy(start);                  \
        cudaEventDestroy(stop);                   \
        printf("GPU Cost Time: %.7f ms\n", time/(iter)); \
    }

#define CPU_Time(EXPER)                                              \
    {                                                                \
        auto start = high_resolution_clock::now();                   \
        (EXPER);                                                     \
        auto stop = high_resolution_clock::now();                    \
        auto duration = duration_cast<microseconds>(stop - start);   \
        printf("CPU Cost Time: %lf ms\n", (double)duration.count()); \
    }

#endif