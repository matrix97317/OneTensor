#ifndef __DATA_TRANSFER_H__
#define __DATA_TRANSFER_H__

#define FLOAT4(ptr) (reinterpret_cast<float4*>(ptr))
#define UINT4(ptr) (reinterpret_cast<uint4*>(ptr))
#define UINT2(ptr) (reinterpret_cast<uint2*>(ptr))
#define FLOAT(ptr) (reinterpret_cast<float*>(ptr))
#define HALF(ptr) (reinterpret_cast<half*>(ptr))

#if ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || (__CUDACC_VER_MAJOR__ > 11)
#define CP_ASYNC_CA(dst, src, Bytes) \
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_CG(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))
#else
#define CP_ASYNC_CA(dst, src, Bytes) \
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_CG(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))
#endif

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)

#define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;\n" ::"n"(N))

#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)


__device__ void load_a_smem_to_reg(half* a, uint32_t* reg, int tid){
    uint32_t *ia0 = (uint32_t*)a;
    reg[0] = ia0[tid];
    reg[1] = ia0[tid+32];
    reg[2] = ia0[tid+64];
    reg[3] = ia0[tid+96];
}

__device__ void load_b_smem_to_reg(half* b, uint32_t* reg,int tid){

    unsigned addr = (unsigned)__cvta_generic_to_shared(&b[tid*8]);
    // uint32_t RC[4];
    asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(reg[0]), "=r"(reg[1]),"=r"(reg[2]), "=r"(reg[3]) : "r"(addr));
}


__device__ void sm_visitor_row(int cols, int group_height,int group_width, int linear_index,int*l1_coord_row, int*l1_coord_col){
    
  
    int group_cols = cols / group_width;
    
    int group_linear_index = linear_index % (group_height*group_width);
    
    
    int group_id = linear_index / (group_height*group_width);
   
    int group_inner_coord_col = group_linear_index %  group_width;
    int group_inner_coord_row = group_linear_index / group_width;
    

    int group_coord_col = group_id % group_cols;
    int group_coord_row = group_id / group_cols;
   
    *l1_coord_col = group_inner_coord_col+group_coord_col*group_width;
    *l1_coord_row = group_inner_coord_row+group_coord_row*group_height;
    
}
__device__ void sm_visitor_col(int rows,int group_height,int group_width,int linear_index, int*l1_coord_row, int*l1_coord_col){
    
    int group_rows = rows / group_height;
  
    
    int group_linear_index = linear_index % (group_height*group_width);
    
    
    int group_id = linear_index / (group_height*group_width);
   
    int group_inner_coord_row = group_linear_index %  group_height;
    int group_inner_coord_col = group_linear_index / group_height;
    
    int group_coord_row = group_id % group_rows;
    int group_coord_col = group_id / group_rows;
   
    *l1_coord_col = group_inner_coord_col+group_coord_col*group_width;
    *l1_coord_row = group_inner_coord_row+group_coord_row*group_height;
}
  
   





#endif
