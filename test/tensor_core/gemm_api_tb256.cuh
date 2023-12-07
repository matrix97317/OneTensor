#ifndef __GEMM_API_H__
#define __GEMM_API_H__

#include "data_compute.cuh"
#include "data_transfer.cuh"

#define MMA_M 16
#define MMA_N 16
#define MMA_K 16

#define STAGE 4

#define WARP_ROW_NUM 4
#define WARP_COL_NUM 2
#define MMA_ROW_NUM 4
#define MMA_COL_NUM 4

#define MMA_ROW_SPLIT 2
#define MMA_COL_SPLIT 4

inline __device__ __host__ size_t div_ceil(size_t a, size_t b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

__device__ void threading_block_compute_order(int *block_row, int *block_col)
{
    // gridDim.y: The number of Row BLOCK.
    // gridDim.x: The number of Col BLOCK.
    // order: U
    *block_row = (blockIdx.z % 2) ? (gridDim.y - blockIdx.y - 1) : blockIdx.y;
    *block_col = (blockIdx.z * gridDim.x + blockIdx.x);
}

__device__ void warp_block_compute_order(int *warp_row, int *warp_col)
{

    int warp_id = threadIdx.x / 32;
    *warp_row = warp_id % WARP_ROW_NUM;
    *warp_col = warp_id / WARP_ROW_NUM;
}

__device__ void a_gmem_offset(int block_row,
                              int warp_row_num,
                              int warp_row,
                              int mma_row_num,
                              int mma_row,
                              int k_id, int K, int mma_m, int mma_k, int lane_id,
                              int *a_offset)
{
    // A offset
    int row = lane_id % 16;
    int col = lane_id / 16;
    int a_row_offset = ((block_row * warp_row_num * mma_row_num + warp_row * (mma_row_num/MMA_ROW_SPLIT) + mma_row) * mma_m);
    int a_col_offset = (k_id * mma_k);
    *a_offset = (a_row_offset + row) * K + col * 8 + a_col_offset;
}

__device__ void b_gmem_offset(
    int block_col,
    int warp_col_num,
    int warp_col,
    int mma_col_num,
    int mma_col,
    int k_id, int N, int mma_n, int mma_k, int lane_id,
    int *b_offset)
{

    // B offset
    int row = lane_id % 16;
    int col = lane_id / 16;
    int b_col_offset = ((block_col * warp_col_num * mma_col_num + warp_col * (mma_col_num/MMA_COL_SPLIT) + mma_col) * mma_n);
    int b_row_offset = (k_id * mma_k);
    *b_offset = (b_row_offset + row) * N + col * 8 + b_col_offset;
}

__device__ void c_gmem_offset(int block_row,
                              int warp_row_num,
                              int warp_row,
                              int mma_row_num,
                              int mma_row,

                              int block_col,
                              int warp_col_num,
                              int warp_col,
                              int mma_col_num,
                              int mma_col,

                              int N, int mma_m, int mma_n, int lane_id,
                              int *c_row0_offset_0,
                              int *c_row0_offset_1,
                              int *c_row1_offset_0,
                              int *c_row1_offset_1)
{

    int row = lane_id / 4;
    int col = lane_id % 4;

    int c_row_offset = ((block_row * warp_row_num * mma_row_num + warp_row * mma_row_num + mma_row) * mma_m);
    int c_col_offset = ((block_col * warp_col_num * mma_col_num + warp_col * mma_col_num + mma_col) * mma_n);
    *c_row0_offset_0 = (c_row_offset + row) * N + col * 2 + c_col_offset;
    *c_row0_offset_1 = (c_row_offset + row) * N + col * 2 + 8 + c_col_offset;

    *c_row1_offset_0 = (c_row_offset + row + 8) * N + col * 2 + c_col_offset;
    *c_row1_offset_1 = (c_row_offset + row + 8) * N + col * 2 + 8 + c_col_offset;
}

__device__ size_t smem_a_offset(
    int warp_row,
    int mma_row_num,
    int mma_row,
    int lane_id,
    int *a_smem_offset)
{
    // A share mem offset
    *a_smem_offset = (warp_row * (mma_row_num/MMA_ROW_SPLIT) + mma_row) * (16 * 16) + lane_id * 8;
}

__device__ size_t smem_b_offset(int warp_col,
                                int mma_col_num,
                                int mma_col,
                                int lane_id,
                                int *b_smem_offset)
{
    // B share mem offset
    *b_smem_offset = (warp_col * (mma_col_num/MMA_COL_SPLIT) + mma_col) * (16 * 16) + lane_id * 8;
}


__global__ void gemm_fp16fp32_tb256(half *a, half *b, float *c,
                                    int M, int N, int K)
{

    // step0: block id
    int block_row = 0;
    int block_col = 0;
    threading_block_compute_order(&block_row, &block_col);
    int m_tiles = div_ceil(M, MMA_M);
    int n_tiles = div_ceil(N, MMA_N);
    int k_tiles = div_ceil(K, MMA_K);
    if ((block_row * WARP_ROW_NUM * MMA_ROW_NUM >= m_tiles) || (block_col * WARP_COL_NUM * MMA_COL_NUM) >= n_tiles)
    {
        return;
    }

    __shared__ half a_smem[1 * STAGE][16 * 16 * WARP_ROW_NUM * MMA_ROW_NUM];
    __shared__ half b_smem[1 * STAGE][16 * 16 * WARP_COL_NUM * MMA_COL_NUM];
    float c_register0[MMA_ROW_NUM][MMA_COL_NUM][4];
    float c_register1[MMA_ROW_NUM][MMA_COL_NUM][4];

    uint32_t a_register[1][MMA_ROW_NUM][4];
    uint32_t b_register[1][MMA_COL_NUM][4];

    // step2: load data to smem
    int warp_id = threadIdx.x / 32;
    int warp_row = 0;
    int warp_col = 0;
    warp_block_compute_order(&warp_row, &warp_col);

    int a_offset = 0;
    int b_offset = 0;
    int c_row0_offset_0 = 0;
    int c_row0_offset_1 = 0;
    int c_row1_offset_0 = 0;
    int c_row1_offset_1 = 0;

    int a_smem_offset = 0;
    int b_smem_offset = 0;
    // int c_smem_offset = 0;
    int lane_id = threadIdx.x % 32;

    int mma_row = 0;
    int mma_col = 0;
   
#pragma unroll
    for (int i = 0; i < MMA_ROW_NUM; i++)
    {
#pragma unroll
        for (int j = 0; j < MMA_COL_NUM; j++)
        {
            mma_row = i;
            mma_col = j; //(i % 2)? (MMA_COL_NUM-j):j;
            c_register0[mma_row][mma_col][0] = 0;
            c_register0[mma_row][mma_col][1] = 0;
            c_register0[mma_row][mma_col][2] = 0;
            c_register0[mma_row][mma_col][3] = 0;

            c_register1[mma_row][mma_col][0] = 0;
            c_register1[mma_row][mma_col][1] = 0;
            c_register1[mma_row][mma_col][2] = 0;
            c_register1[mma_row][mma_col][3] = 0;
        }
    }

    // Load Data Pipeline
    int stage_id = 0;
#pragma unroll
    for (; stage_id < (STAGE - 1); stage_id++)
    {
        // A MMA_ROW_NUM
#pragma unroll
        for (int mma_row_id = 0; mma_row_id < (MMA_ROW_NUM/MMA_ROW_SPLIT); mma_row_id++)
        {
            int a_k_tiles = stage_id*1 + 0;
            smem_a_offset(warp_id,
                         MMA_ROW_NUM,
                         mma_row_id,
                         lane_id,
                         &a_smem_offset);

            a_gmem_offset(block_row,
                          WARP_ROW_NUM,
                          warp_id,
                          MMA_ROW_NUM,
                          mma_row_id,
                          a_k_tiles, K, MMA_M, MMA_K, lane_id,
                          &a_offset);
            CP_ASYNC_CG((uint32_t)__cvta_generic_to_shared(&a_smem[a_k_tiles][a_smem_offset]), &a[a_offset], 16);
        }
        // B MMA_COL_NUM
#pragma unroll
        for (int mma_col_id = 0; mma_col_id < (MMA_COL_NUM/MMA_COL_SPLIT); mma_col_id++)
        {
            int b_k_tiles = stage_id*1 + 0;
            smem_b_offset(warp_id,
                         MMA_COL_NUM,
                         mma_col_id,
                         lane_id,
                         &b_smem_offset);

            b_gmem_offset(block_col,
                          WARP_COL_NUM,
                          warp_id,
                          MMA_COL_NUM,
                          mma_col_id,
                          b_k_tiles, N, MMA_N, MMA_K, lane_id,
                          &b_offset);
            CP_ASYNC_CG((uint32_t)__cvta_generic_to_shared(&b_smem[b_k_tiles][b_smem_offset]), &b[b_offset], 16);
        }
        CP_ASYNC_COMMIT_GROUP();
    }
    // __syncthreads();

    stage_id = 0;
    int save_stage_id = STAGE-2;
#pragma unroll
    for (int k = 1 * (STAGE - 1); k < k_tiles; k = k + 1)
    {

        save_stage_id = (save_stage_id + 1) % STAGE;
#pragma unroll
        for (int mma_row_id = 0; mma_row_id < (MMA_ROW_NUM/MMA_ROW_SPLIT); mma_row_id++)
        {
            smem_a_offset(warp_id,
                         MMA_ROW_NUM,
                         mma_row_id,
                         lane_id,
                         &a_smem_offset);

            a_gmem_offset(block_row,
                          WARP_ROW_NUM,
                          warp_id,
                          MMA_ROW_NUM,
                          mma_row_id,
                          k + 0, K, MMA_M, MMA_K, lane_id,
                          &a_offset);
            CP_ASYNC_CG((uint32_t)__cvta_generic_to_shared(&a_smem[save_stage_id * 1 + 0][a_smem_offset]), &a[a_offset], 16);
        }

#pragma unroll
        for (int mma_col_id = 0; mma_col_id < (MMA_COL_NUM/MMA_COL_SPLIT); mma_col_id++)
        {
            smem_b_offset(warp_id,
                         MMA_COL_NUM,
                         mma_col_id,
                         lane_id,
                         &b_smem_offset);

            b_gmem_offset(block_col,
                          WARP_COL_NUM,
                          warp_id,
                          MMA_COL_NUM,
                          mma_col_id,
                          k + 0, N, MMA_N, MMA_K, lane_id,
                          &b_offset);
            CP_ASYNC_CG((uint32_t)__cvta_generic_to_shared(&b_smem[save_stage_id * 1+0][b_smem_offset]), &b[b_offset], 16);
        }

        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP((((STAGE-2)>0)?(STAGE-2):1));
        

        // __syncthreads(); //stage 1

        // load to a register
#pragma unroll
        for(int i=0; i<MMA_ROW_NUM;i++){
            mma_row = i;
            load_a_smem_to_reg(&a_smem[0 + stage_id * 1][(warp_row * MMA_ROW_NUM + mma_row) * (16 * 16)], a_register[0][mma_row], lane_id);
            // load_a_smem_to_reg(&a_smem[1 + stage_id * 2][(warp_row * MMA_ROW_NUM + mma_row) * (16 * 16)], a_register[1][mma_row], lane_id);
        }
        // load to b register
#pragma unroll
        for(int j=0; j<MMA_COL_NUM;j++){
            mma_col=j;
            load_b_smem_to_reg(&b_smem[0 + stage_id * 1][(warp_col * MMA_COL_NUM + mma_col) * (16 * 16)], b_register[0][mma_col],lane_id);
            // load_b_smem_to_reg(&b_smem[1 + stage_id * 2][(warp_col * MMA_COL_NUM + mma_col) * (16 * 16)], b_register[1][mma_col],lane_id);
        }

        

#pragma unroll
        for (int i = 0; i < MMA_ROW_NUM; i++)
        {
#pragma unroll
            for (int j = 0; j < MMA_COL_NUM; j++)
            {
                mma_row = i;
                mma_col = (i % 2) ? (MMA_COL_NUM - j - 1) : j;
                mma_m16n8k16_fp16fp32_v3(a_register[0][mma_row],
                                         b_register[0][mma_col],
                                         c_register0[mma_row][mma_col],
                                         c_register0[mma_row][mma_col],
                                         c_register1[mma_row][mma_col],
                                         c_register1[mma_row][mma_col]);
                // __syncthreads();
                //  mma_m16n8k16_fp16fp32_v3(a_register[1][mma_row],
                //                          b_register[1][mma_col],
                //                          c_register0[mma_row][mma_col],
                //                          c_register0[mma_row][mma_col],
                //                          c_register1[mma_row][mma_col],
                //                          c_register1[mma_row][mma_col]);
              
                // __syncthreads();
            }
        }
        stage_id = (stage_id + 1) % STAGE;
    }
    CP_ASYNC_WAIT_GROUP(0);
    // __syncthreads();
#pragma unroll
    for (int tail = 0; tail < (STAGE-1); tail++)
    {
        int tail_stage_id = (stage_id + tail) % STAGE;


        // load to a register
#pragma unroll
        for(int i=0; i<MMA_ROW_NUM;i++){
            mma_row = i;
            load_a_smem_to_reg(&a_smem[0 + tail_stage_id * 1][(warp_row * MMA_ROW_NUM + mma_row) * (16 * 16)], a_register[0][mma_row], lane_id);
            // load_a_smem_to_reg(&a_smem[1 + tail_stage_id * 2][(warp_row * MMA_ROW_NUM + mma_row) * (16 * 16)], a_register[1][mma_row], lane_id);
        }
        // load to b register
#pragma unroll
        for(int j=0; j<MMA_COL_NUM;j++){
            mma_col=j;
            load_b_smem_to_reg(&b_smem[0 + tail_stage_id * 1][(warp_col * MMA_COL_NUM + mma_col) * (16 * 16)], b_register[0][mma_col],lane_id);
            // load_b_smem_to_reg(&b_smem[1 + tail_stage_id * 2][(warp_col * MMA_COL_NUM + mma_col) * (16 * 16)], b_register[1][mma_col],lane_id);
        }





#pragma unroll
        for (int i = 0; i < MMA_ROW_NUM; i++)
        {
#pragma unroll
            for (int j = 0; j < MMA_COL_NUM; j++)
            {
                mma_row = i;
                mma_col = (i % 2) ? (MMA_COL_NUM - j - 1) : j;
                mma_m16n8k16_fp16fp32_v3(a_register[0][mma_row],
                                         b_register[0][mma_col],
                                         c_register0[mma_row][mma_col],
                                         c_register0[mma_row][mma_col],
                                         c_register1[mma_row][mma_col],
                                         c_register1[mma_row][mma_col]);
                // __syncthreads();
                // mma_m16n8k16_fp16fp32_v3(a_register[1][mma_row],
                //                          b_register[1][mma_col],
                //                          c_register0[mma_row][mma_col],
                //                          c_register0[mma_row][mma_col],
                //                          c_register1[mma_row][mma_col],
                //                          c_register1[mma_row][mma_col]);
                // __syncthreads();
            }
        }
    }
    
#pragma unroll
    for (int i = 0; i < MMA_ROW_NUM; i++)
    {
#pragma unroll
        for (int j = 0; j < MMA_COL_NUM; j++)
        {
            mma_row = i;
            mma_col = (i % 2) ? (MMA_COL_NUM - j - 1) : j;
            c_gmem_offset(block_row,
                          WARP_ROW_NUM,
                          warp_row,
                          MMA_ROW_NUM,
                          mma_row,
                          block_col,
                          WARP_COL_NUM,
                          warp_col,
                          MMA_COL_NUM,
                          mma_col,
                          N, MMA_M, MMA_N, lane_id,
                          &c_row0_offset_0,
                          &c_row0_offset_1,
                          &c_row1_offset_0,
                          &c_row1_offset_1);

            UINT2(&c[c_row0_offset_0])[0] = UINT2(&c_register0[mma_row][mma_col][0])[0];
            UINT2(&c[c_row1_offset_0])[0] = UINT2(&c_register0[mma_row][mma_col][2])[0];

            UINT2(&c[c_row0_offset_1])[0] = UINT2(&c_register1[mma_row][mma_col][0])[0];
            UINT2(&c[c_row1_offset_1])[0] = UINT2(&c_register1[mma_row][mma_col][2])[0];
        }
    }
}
#endif