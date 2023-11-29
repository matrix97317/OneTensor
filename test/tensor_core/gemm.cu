#include <cuda_runtime.h>
#include <iostream>
#include <one_tensor.h>
#include <cublas_v2.h>
#include "data_transfer.cuh"
#include "data_compute.cuh"
using namespace std;

// #define FLOAT4(ptr) (reinterpret_cast<float4*>(ptr))
// #define UINT4(ptr) (reinterpret_cast<uint4*>(ptr))
// #define FLOAT(ptr) (reinterpret_cast<float*>(ptr))
// #define HALF(ptr) (reinterpret_cast<half*>(ptr))

// __device__ void load_block_major_row(float* global_ptr, 
//                                         float* out, 
//                                         size_t row_block_index, 
//                                         size_t col_block_index, 
//                                         size_t rows, 
//                                         size_t cols, 
//                                         size_t block_row, 
//                                         size_t block_col,
//                                         size_t warp_lane_id,
//                                         size_t group_size=1
//                                         ){
        
//         size_t global_start_row = row_block_index*block_row;
//         size_t global_start_col = col_block_index*block_col;
//         // 1D -> 2D
//         size_t local_row = warp_lane_id / (block_col/group_size);
//         size_t local_col = warp_lane_id % (block_col/group_size);
//         // 2D -> 1D(global)
//         // mask
//         size_t global_col = global_start_col+local_col * group_size;
//         size_t global_row = global_start_row+local_row * 1;
        
        
//         if (group_size==1){
//             if ((global_col >= cols) || (global_row >= rows)){
//                 out[0] = 0;
//             }else{
//                 size_t global_offset = global_row*cols+global_col;
//                 out[0] = global_ptr[global_offset];
//             }
//         }

//         if (group_size==4){
//             size_t global_offset = global_row*cols+global_col;
//             if (global_row >= rows){
//                 out[0] = 0;
//                 out[1] = 0;
//                 out[2] = 0;
//                 out[3] = 0;
//             }
//             else if ((global_col+group_size >= cols) && (global_row < rows)){
//                 for(int i=0; i<(cols-global_col);i++){
//                     out[i]=global_ptr[global_offset+i];
//                 }
//             }else{
//                 float4 data = FLOAT4(&global_ptr[global_offset])[0];
//                 out[0] = FLOAT(&data)[0];
//                 out[1] = FLOAT(&data)[1];
//                 out[2] = FLOAT(&data)[2];
//                 out[3] = FLOAT(&data)[3];
//             }
//         }
// }

// __device__ void store_block_major_row(float* global_ptr, 
//                                         float* in, 
//                                         size_t row_block_index, 
//                                         size_t col_block_index, 
//                                         size_t rows, 
//                                         size_t cols, 
//                                         size_t block_row, 
//                                         size_t block_col,
//                                         size_t warp_lane_id,
//                                         size_t group_size=1
//                                         ){
        
//         size_t global_start_row = row_block_index*block_row;
//         size_t global_start_col = col_block_index*block_col;
//         // 1D -> 2D
//         size_t local_row = warp_lane_id / (block_col/group_size);
//         size_t local_col = warp_lane_id % (block_col/group_size);
//         // 2D -> 1D(global)
//         // mask
//         size_t global_col = global_start_col+local_col * group_size;
//         size_t global_row = global_start_row+local_row * 1;
        
//         if (group_size==1){
//             if ((global_col >= cols) || (global_row >= rows)){
//                 ;
//             }
//             else{
//                 size_t global_offset = global_row*cols+global_col;
//                 global_ptr[global_offset] = in[0];
//             }
//         }
       
//         if (group_size==4){
//             size_t global_offset = global_row*cols+global_col;
//             if (global_row >=rows){
//                 ;
//             }else if((global_col+group_size >= cols) && (global_row < rows)){
//                 for(int i=0; i<(cols-global_col);i++){
//                     global_ptr[global_offset+i]=in[i];
//                 }
//             }else{
//                 float4 data;
//                 FLOAT(&data)[0] = in[0];
//                 FLOAT(&data)[1] = in[1];
//                 FLOAT(&data)[2] = in[2];
//                 FLOAT(&data)[3] = in[3];
//                 FLOAT4(&global_ptr[global_offset])[0]=data;
//             }
//         }
// }

// __device__ void load_block_major_col(float* global_ptr, 
//                                         float* out, 
//                                         size_t row_block_index, 
//                                         size_t col_block_index, 
//                                         size_t rows, 
//                                         size_t cols, 
//                                         size_t block_row, 
//                                         size_t block_col,
//                                         size_t warp_lane_id,
//                                         size_t group_size=1
//                                         ){
        
//         size_t global_start_row = row_block_index*block_row;
//         size_t global_start_col = col_block_index*block_col;
//         // 1D -> 2D
//         size_t local_row = warp_lane_id % (block_row/group_size);
//         size_t local_col = warp_lane_id / (block_row/group_size);
//         // 2D -> 1D(global)
//         // mask
//         size_t global_col = global_start_col+local_col * 1;
//         size_t global_row = global_start_row+local_row * group_size;
        
//         if(group_size==1){
//             if ((global_col >= cols) || (global_row >= rows)){
//                 out[0] = 0;
//             }else{
//                 size_t global_offset = global_row*cols+global_col;
//                 out[0] = global_ptr[global_offset];
//             }
//         }

//         if(group_size==4){
//             size_t global_offset = global_row*cols+global_col;
//             if ((global_col >= cols)){
//                 out[0] = 0;
//                 out[1] = 0;
//                 out[2] = 0;
//                 out[3] = 0;
//             }else if ((global_col < cols) && (global_row+group_size >= rows)){
//                 for(int i=0; i<(rows-global_row);i++){
//                     out[i] = global_ptr[global_offset+i*cols];
//                 }
//             }else{
//                 out[0] = global_ptr[global_offset+0*cols];
//                 out[1] = global_ptr[global_offset+1*cols];
//                 out[2] = global_ptr[global_offset+2*cols];
//                 out[3] = global_ptr[global_offset+3*cols];
//             }
//         }
// }

// __device__ void load_block_major_row_half(half* global_ptr, 
//                                         half* out, 
//                                         size_t row_block_index, 
//                                         size_t col_block_index, 
//                                         size_t rows, 
//                                         size_t cols, 
//                                         size_t block_row, 
//                                         size_t block_col,
//                                         size_t warp_lane_id,
//                                         size_t group_size=1
//                                         ){
        
//         size_t global_start_row = row_block_index*block_row;
//         size_t global_start_col = col_block_index*block_col;
//         // 1D -> 2D
//         size_t local_row = warp_lane_id / (block_col/group_size);
//         size_t local_col = warp_lane_id % (block_col/group_size);
//         // 2D -> 1D(global)
//         // mask
//         size_t global_col = global_start_col+local_col * group_size;
//         size_t global_row = global_start_row+local_row * 1;
        
        
//         if (group_size==1){
//             if ((global_col >= cols) || (global_row >= rows)){
//                 out[0] = 0;
//             }else{
//                 size_t global_offset = global_row*cols+global_col;
//                 out[0] = global_ptr[global_offset];
//             }
//         }

//         if (group_size==8){
//             size_t global_offset = global_row*cols+global_col;
//             if (global_row >= rows){
//                 out[0] = 0;
//                 out[1] = 0;
//                 out[2] = 0;
//                 out[3] = 0;
//                 out[4] = 0;
//                 out[5] = 0;
//                 out[6] = 0;
//                 out[7] = 0;

//             }
//             else if ((global_col+group_size >= cols) && (global_row < rows)){
//                 for(int i=0; i<(cols-global_col);i++){
//                     out[i]=global_ptr[global_offset+i];
//                 }
//             }else{

//                 uint4 data = UINT4(&global_ptr[global_offset])[0];
//                 out[0] = HALF(&data)[0];
//                 out[1] = HALF(&data)[1];
//                 out[2] = HALF(&data)[2];
//                 out[3] = HALF(&data)[3];
//                 out[4] = HALF(&data)[4];
//                 out[5] = HALF(&data)[5];
//                 out[6] = HALF(&data)[6];
//                 out[7] = HALF(&data)[7];
//             }
//         }
// }

// __device__ void store_block_major_row_half(half* global_ptr, 
//                                         half* in, 
//                                         size_t row_block_index, 
//                                         size_t col_block_index, 
//                                         size_t rows, 
//                                         size_t cols, 
//                                         size_t block_row, 
//                                         size_t block_col,
//                                         size_t warp_lane_id,
//                                         size_t group_size=1
//                                         ){
        
//         size_t global_start_row = row_block_index*block_row;
//         size_t global_start_col = col_block_index*block_col;
//         // 1D -> 2D
//         size_t local_row = warp_lane_id / (block_col/group_size);
//         size_t local_col = warp_lane_id % (block_col/group_size);
//         // 2D -> 1D(global)
//         // mask
//         size_t global_col = global_start_col+local_col * group_size;
//         size_t global_row = global_start_row+local_row * 1;
        
//         if (group_size==1){
//             if ((global_col >= cols) || (global_row >= rows)){
//                 ;
//             }
//             else{
//                 size_t global_offset = global_row*cols+global_col;
//                 global_ptr[global_offset] = in[0];
//             }
//         }
       
//         if (group_size==8){
//             size_t global_offset = global_row*cols+global_col;
//             if (global_row >=rows){
//                 ;
//             }else if((global_col+group_size >= cols) && (global_row < rows)){
//                 for(int i=0; i<(cols-global_col);i++){
//                     global_ptr[global_offset+i]=in[i];
//                 }
//             }else{
//                 uint4 data;
//                 HALF(&data)[0] = in[0];
//                 HALF(&data)[1] = in[1];
//                 HALF(&data)[2] = in[2];
//                 HALF(&data)[3] = in[3];
//                 HALF(&data)[4] = in[4];
//                 HALF(&data)[5] = in[5];
//                 HALF(&data)[6] = in[6];
//                 HALF(&data)[7] = in[7];
//                 UINT4(&global_ptr[global_offset])[0]=data;
//             }
//         }
// }

// __device__ void load_block_major_col_half(half* global_ptr, 
//                                         half* out, 
//                                         size_t row_block_index, 
//                                         size_t col_block_index, 
//                                         size_t rows, 
//                                         size_t cols, 
//                                         size_t block_row, 
//                                         size_t block_col,
//                                         size_t warp_lane_id,
//                                         size_t group_size=1
//                                         ){
        
//         size_t global_start_row = row_block_index*block_row;
//         size_t global_start_col = col_block_index*block_col;
//         // 1D -> 2D
//         size_t local_row = warp_lane_id % (block_row/group_size);
//         size_t local_col = warp_lane_id / (block_row/group_size);
//         // 2D -> 1D(global)
//         // mask
//         size_t global_col = global_start_col+local_col * 1;
//         size_t global_row = global_start_row+local_row * group_size;
        
//         if(group_size==1){
//             if ((global_col >= cols) || (global_row >= rows)){
//                 out[0] = 0;
//             }else{
//                 size_t global_offset = global_row*cols+global_col;
//                 out[0] = global_ptr[global_offset];
//             }
//         }

//         if(group_size==8){
//             size_t global_offset = global_row*cols+global_col;
//             if ((global_col >= cols)){
//                 out[0] = 0;
//                 out[1] = 0;
//                 out[2] = 0;
//                 out[3] = 0;
//                 out[4] = 0;
//                 out[5] = 0;
//                 out[6] = 0;
//                 out[7] = 0;
//             }else if ((global_col < cols) && (global_row+group_size >= rows)){
//                 for(int i=0; i<(rows-global_row);i++){
//                     out[i] = global_ptr[global_offset+i*cols];
//                 }
//             }else{
//                 out[0] = global_ptr[global_offset+0*cols];
//                 out[1] = global_ptr[global_offset+1*cols];
//                 out[2] = global_ptr[global_offset+2*cols];
//                 out[3] = global_ptr[global_offset+3*cols];
//                 out[4] = global_ptr[global_offset+4*cols];
//                 out[5] = global_ptr[global_offset+5*cols];
//                 out[6] = global_ptr[global_offset+6*cols];
//                 out[7] = global_ptr[global_offset+7*cols];
//             }
//         }
// }



__global__ void gemm_tb32_fp16fp32(
                     half* mat_a, 
                     half* mat_b, 
                     float* mat_c,
                     size_t M,
                     size_t N,
                     size_t K,
                     size_t warp_m,
                     size_t warp_n,
                     size_t warp_k,
                     size_t warp_x_num_in_tb,
                     size_t warp_y_num_in_tb
                     ){
    
    int bid = blockIdx.y * gridDim.x + blockIdx.x;
    int tid = threadIdx.x;
    // int warp_id = tid / 32;
    int lane_id = tid % 32;
    // step1. load data
    __shared__ half mat_a_smem[2][8*32];
    __shared__ half mat_b_smem[2][4*32];
    // __shared__ float mat_c_smem[16*8];
    __shared__ float mat_d_smem[16*8];
    // __shared__ float mat_e_smem[16*8];
    int sm_row_id=0;
    int sm_col_id=0;
    sm_visitor_col(gridDim.x,1,1,bid, &sm_row_id, &sm_col_id);
    // printf("bid %d row %d col %d\n",bid,sm_row_id,sm_col_id);
    // printf("warp_x_num_in_tb row %d\n",warp_x_num_in_tb);
    // printf("sm col %d\n",sm_col_id);
   
    mat_d_smem[lane_id*4+0]=0;
    mat_d_smem[lane_id*4+1]=0;
    mat_d_smem[lane_id*4+2]=0;
    mat_d_smem[lane_id*4+3]=0;
    int double_buffer=0;
    load_block_major_row_half(mat_a,
                                &mat_a_smem[double_buffer][lane_id*8],
                                sm_row_id*warp_y_num_in_tb+0, 
                                0,
                                M,
                                K,
                                warp_m, 
                                warp_k, 
                                lane_id,
                                8);
    load_block_major_col_half(mat_b,
                                &mat_b_smem[double_buffer][lane_id*4],
                                0,
                                sm_col_id*warp_x_num_in_tb+0, 
                                K,
                                N,
                                warp_k, 
                                warp_n, 
                                lane_id,
                                4);
    __syncthreads();
    for(int k=1; k<(K/warp_k); k++){
        mma_m16n8k16_fp16fp32(mat_a_smem[double_buffer], mat_b_smem[double_buffer], mat_d_smem, mat_d_smem,lane_id);
        double_buffer = double_buffer^1;
        load_block_major_row_half(mat_a,
                                &mat_a_smem[double_buffer][lane_id*8],
                                sm_row_id*warp_y_num_in_tb+0, 
                                k,
                                M,
                                K,
                                warp_m, 
                                warp_k, 
                                lane_id,
                                8);
        load_block_major_col_half(mat_b,
                                &mat_b_smem[double_buffer][lane_id*4],
                                k,
                                sm_col_id*warp_x_num_in_tb+0, 
                                K,
                                N,
                                warp_k, 
                                warp_n, 
                                lane_id,
                                4);
        __syncthreads();
        // step2. compute data
    }
    mma_m16n8k16_fp16fp32(mat_a_smem[double_buffer], mat_b_smem[double_buffer], mat_d_smem, mat_d_smem,lane_id);
    // step3. store data
    store_block_major_row(mat_c,
                              &mat_d_smem[lane_id*4],
                              sm_row_id*warp_y_num_in_tb+0,
                              sm_col_id*warp_x_num_in_tb+0, 
                              M,
                              N,
                              warp_m,
                              warp_n,
                              lane_id,
                              4);
}

int main(){

    int measure_times = 200;
    OneTensor<float> flops(Shape(measure_times));
    OneTensor<float> cost_time(Shape(measure_times));
    for(int cnt=199; cnt<measure_times;cnt++){
    
    int M = 16*(cnt+1)*5;
    int N = 8*(cnt+1)*5;
    int K = 16*1;
    int warp_m = 16;
    int warp_n = 8;
    int warp_k = 16;

    OneTensor<half> a(Shape(M,K));
    OneTensor<half> b(Shape(K,N));
    OneTensor<float> c(Shape(M,N));
    c.FillHostData(0);
    for(int i=0; i<a.shape.d0; i++){
        for(int j=0; j<a.shape.d1; j++){
            int index = i*a.shape.d1+j;
            a.SetHostData(index*1,index);
        }
    }
    for(int i=0; i<b.shape.d0; i++){
        for(int j=0; j<b.shape.d1; j++){
            int index = i*b.shape.d1+j;
            b.SetHostData(2,index);
        }
    }

    // a.HostDataView();
    // b.HostDataView();
    // c.HostDataView();
 
    a.sync_device();
    b.sync_device();
    c.sync_device();
    
   
    dim3 gird(ceil(N/warp_n),ceil(M/warp_m) ,1);
    dim3 block(32,1,1);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // gemm_tb32_fp16fp32<<<gird,block,0,stream>>>(a.deviceData<half>(),
    //                                             b.deviceData<half>(),
    //                                             c.deviceData<float>(),
    //                                             M,
    //                                             N,
    //                                             K,
    //                                             warp_m,
    //                                             warp_n,
    //                                             warp_k,
    //                                             1,
    //                                             1);
    // GPU_Time((gemm_tb32_fp16fp32<<<gird,block,0,stream>>>(a.deviceData<half>(),
    //                                             b.deviceData<half>(),
    //                                             c.deviceData<float>(),
    //                                             M,
    //                                             N,
    //                                             K,
    //                                             warp_m,
    //                                             warp_n,
    //                                             warp_k,
    //                                             1,
    //                                             1)), stream, 200,0);
    // GPU_Time((ld_block<<<gird,block,0,stream>>>(a.deviceData<float>(),b.deviceData<float>())), 
    //             stream,
    //             10,
    //             Shape(8,4).size<float>()*2);
    

    float alpha = 1.0;
    float beta = 0.0;
    cudaDeviceSynchronize();
    cublasHandle_t blas_handle;
    cublasCreate(&blas_handle);
    cublasSetStream(blas_handle, stream);
    // (cublasHgemm(blas_handle,
    //                     CUBLAS_OP_N,
    //                     CUBLAS_OP_N,
    //                     N,
    //                     M,
    //                     K,
    //                     &alpha,
    //                     b.deviceData<half>(),
    //                     N,
    //                     a.deviceData<half>(),
    //                     K,
    //                     &beta,
    //                     c.deviceData<half>(),
    //                     N));
    // GPU_Time((cublasHgemm(blas_handle,
    //                     CUBLAS_OP_N,
    //                     CUBLAS_OP_N,
    //                     N,
    //                     M,
    //                     K,
    //                     &alpha,
    //                     b.deviceData<half>(),
    //                     N,
    //                     a.deviceData<half>(),
    //                     K,
    //                     &beta,
    //                     c.deviceData<half>(),
    //                     N)),stream,200,0);
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
    (cublasGemmEx(
        blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
        &alpha, b.deviceData<half>(), CUDA_R_16F, N, a.deviceData<half>(), CUDA_R_16F, K, &beta, c.deviceData<float>(), CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F, algo_list[1]));
    // GPU_Time((cublasGemmEx(
    //     blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
    //     &alpha, b.deviceData<half>(), CUDA_R_16F, N, a.deviceData<half>(), CUDA_R_16F, K, &beta, c.deviceData<float>(), CUDA_R_32F, N,
    //     CUBLAS_COMPUTE_32F, algo_list[1])),stream,200,0);
    cublasDestroy(blas_handle);

 
    // c.sync_device(false);
    // c.HostDataView();
    
    // cost_time.SetHostData((mtime/(200)),cnt);
    // flops.SetHostData(M*N*K,cnt);
    // a.release();
    // b.release();
    // c.release();
    cudaStreamDestroy(stream);
    }
    // cost_time.SaveNpyFile<float>("cublas_cost_time.npy");
    // flops.SaveNpyFile<float>("cublas_flops.npy");

    
    return 0;
}