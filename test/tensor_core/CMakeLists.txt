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
include_directories("/usr/local/cuda/include") # cuda include dir
include_directories("./") # current include dir
# set lib file
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
    load_block
    "load_block.cu"
    )
# set link lib for executed file
target_link_libraries(
    load_block
    # vec_add_shared
    -lcuda
    -lcudart
    -lcublas
    -lone_tensor
    )