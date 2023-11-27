#include <cuda_runtime.h>
#include <iostream>
#include <one_tensor.h>
#include "data_transfer.cuh"
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



__global__ void ld_block(half  *a, half  *b){

    

    int tid = blockDim.x*blockIdx.x+threadIdx.x; // wrap: 32Thread.
   
    __shared__ half smem[8*4*8];
    
    load_block_major_col_half(a,&smem[tid*8], 0,1, 32*8, 32,  32*8,1, tid,8);
    __syncthreads();
    store_block_major_row_half(b,&smem[tid*8], 1,0, 2,  32*8,  1, 32*8, tid,8);
    
    // // b[tid]=smem[tid];
    // if (tid < 31){
    // b[tid*4+0]=smem[tid*4+0];
    // b[tid*4+1]=smem[tid*4+1];
    // b[tid*4+2]=smem[tid*4+2];
    // b[tid*4+3]=smem[tid*4+3];
    // }
}

int main(){
    OneTensor<half> a(Shape(32*8,32));
    OneTensor<half> b(Shape(2,32*8));
    b.FillHostData(0);
    for(int i=0; i<a.shape.d0; i++){
        for(int j=0; j<a.shape.d1; j++){
            int index = i*a.shape.d1+j;
            a.SetHostData(half(index*1.0),index);
            // b.SetHostData(j*1.0,index);
        }
    }
    a.HostDataView();
    // b.HostDataView();
 
    a.sync_device();
    b.sync_device();
    
   
    dim3 gird(1,1,1);
    dim3 block(32,1,1);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    ld_block<<<gird,block,0,stream>>>(a.deviceData<half>(),
                                       b.deviceData<half>()
                                       );
    // GPU_Time((ld_block<<<gird,block,0,stream>>>(a.deviceData<float>(),b.deviceData<float>())), 
    //             stream,
    //             10,
    //             Shape(8,4).size<float>()*2);
   
    b.sync_device(false);
    b.HostDataView();
    cudaStreamDestroy(stream);
    return 0;
}