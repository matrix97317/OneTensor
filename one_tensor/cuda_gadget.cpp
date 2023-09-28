#include "cuda_gadget.h"

const char *getCublasErrorName(cublasStatus_t err) {
    const char *error = "";
    switch (err) {
    case cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED:
        error = "CUBLAS_STATUS_NOT_INITIALIZED";
        break;
    case cublasStatus_t::CUBLAS_STATUS_ALLOC_FAILED:
        error = "CUBLAS_STATUS_ALLOC_FAILED";
        break;
    case cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE:
        error = "CUBLAS_STATUS_INVALID_VALUE";
        return error;
        break;
    case cublasStatus_t::CUBLAS_STATUS_ARCH_MISMATCH:
        error = "CUBLAS_STATUS_ARCH_MISMATCH";
        break;
    case cublasStatus_t::CUBLAS_STATUS_MAPPING_ERROR:
        error = "CUBLAS_STATUS_MAPPING_ERROR";
        break;
    case cublasStatus_t::CUBLAS_STATUS_EXECUTION_FAILED:
        error = "CUBLAS_STATUS_EXECUTION_FAILED";
        break;
    case cublasStatus_t::CUBLAS_STATUS_INTERNAL_ERROR:
        error = "CUBLAS_STATUS_INTERNAL_ERROR";
        break;
    case cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED:
        error = "CUBLAS_STATUS_NOT_SUPPORTED";
        break;
    case cublasStatus_t::CUBLAS_STATUS_LICENSE_ERROR:
        error = "CUBLAS_STATUS_LICENSE_ERROR";
        break;
    default:
        error = "unknown cublas error";
        break;
    }
    return error;
}