cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(one_tensor LANGUAGES CXX CUDA)

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
                    "-Xptxas -dlcm=cv"
                    "-Xptxas -dscm=wt")
                    # "--gencode arch=compute_86,code=sm_86")

message(STATUS "CUDA_NVCC_FLAGS = ${CUDA_NVCC_FLAGS}")

# set include file
include_directories("/usr/local/cuda/include") # cuda include dir
# set lib file
link_directories("/usr/local/cuda/lib64")
# set lib source file
file(GLOB_RECURSE LIB_SRC "${CMAKE_SOURCE_DIR}/cuda_gadget.cpp")
message(STATUS "LIB_SRC = ${LIB_SRC}")

# build shared lib
add_library(one_tensor SHARED ${LIB_SRC})
target_include_directories(one_tensor PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(one_tensor PROPERTIES PUBLIC_HEADER 
"${CMAKE_SOURCE_DIR}/include/cuda_gadget.h;\
${CMAKE_SOURCE_DIR}/include/one_tensor.h")

install(
    TARGETS one_tensor
    EXPORT MyOneTensorTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    PUBLIC_HEADER DESTINATION include
    )

