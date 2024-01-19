cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(vec_add LANGUAGES CXX CUDA)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# cuda packages
find_package(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
set(CUDA_ARCHITECTURE_FLAGS "80")
# set nvcc's options
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
                    "-U__CUDA_NO_HALF_OPERATORS__"
                    "-U__CUDA_NO_HALF_CONVERSIONS__"
                    "-U__CUDA_NO_HALF2_OPERATORS__"
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__"
                    "--expt-relaxed-constexpr"
                    "--expt-extended-lambda"
                    "--use_fast_math"
                    # "-Xptxas -dlcm=cv"
                    # "-Xptxas -dscm=wt"
                    "-Xptxas -dlcm=cg")
                    # "--gencode arch=compute_86,code=sm_86")

message(STATUS "CUDA_NVCC_FLAGS = ${CUDA_NVCC_FLAGS}")

# set include file
include_directories("../cutlass/include")
include_directories("../cutlass/tools/util/include")
include_directories("../cutlass/tools/library/include")
include_directories("/usr/local/cuda/include") # cuda include dir
include_directories("/workspace/code_space/OneTensor/libcusparse_lt-linux-x86_64-0.4.0.7-archive/include")
include_directories("./") # current include dir
# set lib file
link_directories("/workspace/code_space/OneTensor/libcusparse_lt-linux-x86_64-0.4.0.7-archive/lib")
link_directories("/usr/local/cuda/lib64")
# set lib source file
# file(GLOB_RECURSE LIB_SRC "./*.cu")
# message(STATUS "LIB_SRC = ${LIB_SRC}")

# build shared lib
# add_library(vec_add_shared SHARED ${LIB_SRC})
# build executed file
add_executable(
    ldmatrix
    "ldmatrix.cu"
    )
# set link lib for executed file
target_link_libraries(
    ldmatrix
    # vec_add_shared
    -lcuda
    -lcudart
    -lone_tensor
    )

add_executable(
    mma_m8n8k16_s8s8s32
    "mma_m8n8k16_s8s8s32.cu"
    )
# set link lib for executed file
target_link_libraries(
    mma_m8n8k16_s8s8s32
    # vec_add_shared
    -lcuda
    -lcudart
    -lone_tensor
    )

add_executable(
    mma_m16n8k8_f16f16f32
    "mma_m16n8k8_f16f16f32.cu"
    )
# set link lib for executed file
target_link_libraries(
    mma_m16n8k8_f16f16f32
    # vec_add_shared
    -lcuda
    -lcudart
    -lone_tensor
    )

add_executable(
    mma_m16n8k16_f16f16f16
    "mma_m16n8k16_f16f16f16.cu"
    )
# set link lib for executed file
target_link_libraries(
    mma_m16n8k16_f16f16f16
    # vec_add_shared
    -lcuda
    -lcudart
    -lone_tensor
    )

add_executable(
    mma_m16n8k16_f16f16f32
    "mma_m16n8k16_f16f16f32.cu"
    )
# set link lib for executed file
target_link_libraries(
    mma_m16n8k16_f16f16f32
    # vec_add_shared
    -lcuda
    -lcudart
    -lone_tensor
    )

add_executable(
    mma_m16n8k32_s8s8s32
    "mma_m16n8k32_s8s8s32.cu"
    )
# set link lib for executed file
target_link_libraries(
    mma_m16n8k32_s8s8s32
    # vec_add_shared
    -lcuda
    -lcudart
    -lone_tensor
    )

add_executable(
    mma_m16n8k64_s4s4s32
    "mma_m16n8k64_s4s4s32.cu"
    )
# set link lib for executed file
target_link_libraries(
    mma_m16n8k64_s4s4s32
    # vec_add_shared
    -lcuda
    -lcudart
    -lone_tensor
    )

add_executable(
    gemm
    "gemm.cu"
    )
# set link lib for executed file
target_link_libraries(
    gemm
    # vec_add_shared
    -lcuda
    -lcudart
    -lcublas
    -lone_tensor
    )
add_executable(
    gemm_split_k
    "gemm_split_k.cu"
    )
# set link lib for executed file
target_link_libraries(
    gemm_split_k
    # vec_add_shared
    -lcuda
    -lcudart
    -lcublas
    -lone_tensor
    )
add_executable(
    sp_mma_m16n8k16_f16f16f32
    "sp_mma_m16n8k16_f16f16f32.cu"
    )
# set link lib for executed file
target_link_libraries(
    sp_mma_m16n8k16_f16f16f32
    # vec_add_shared
    -lcuda
    -lcudart
    -lone_tensor
    )
  
add_executable(
    conv2d
    "conv2d.cu"
    )
# set link lib for executed file
target_link_libraries(
    conv2d
    # vec_add_shared
    -lcudnn
    -lcuda
    -lcudart
    -lone_tensor
    )

add_executable(
    reduce
    "reduce.cu"
    )
# set link lib for executed file
target_link_libraries(
    reduce
    # vec_add_shared
    -lcudnn
    -lcuda
    -lcudart
    -lone_tensor
    )

add_executable(
    cusparse_lt
    "cusparse_lt.cu"
    )
# set link lib for executed file
target_link_libraries(
    cusparse_lt
    # vec_add_shared
    -lcudnn
    -lcuda
    -lcudart
    -lcusparse
    -lcusparseLt
    -lone_tensor
    )


add_executable(
    cusparse_lt2
    "cusparse_lt2.cu"
    )
# set link lib for executed file
target_link_libraries(
    cusparse_lt2
    # vec_add_shared
    -lcudnn
    -lcuda
    -lcudart
    -lcusparse
    -lcusparseLt
    -lone_tensor
    )

add_executable(
    sparse_conv2d
    "sparse_conv2d.cu"
    )
# set link lib for executed file
target_link_libraries(
    sparse_conv2d    
    # vec_add_shared
    -lcudnn
    -lcuda
    -lcublas
    -lcudart
    -lcusparse
    -lcusparseLt
    -lone_tensor
    )

add_executable(
    gemm_cute
    "gemm_cute.cu"
    )
# set link lib for executed file
target_link_libraries(
    gemm_cute    
    # vec_add_shared
    -lcudnn
    -lcuda
    -lcublas
    -lcudart
    -lcusparse
    -lcusparseLt
    -lone_tensor
    )

add_executable(
    layer_norm
    "layer_norm.cu"
    )
# set link lib for executed file
target_link_libraries(
    layer_norm    
    # vec_add_shared
    -lcudnn
    -lcuda
    -lcublas
    -lcudart
    -lcusparse
    -lcusparseLt
    -lone_tensor
    )