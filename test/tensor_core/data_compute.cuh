#ifndef __DATA_COMPUTE_H__
#define __DATA_COMPUTE_H__

__device__ void mma_m16n8k16_fp16fp32(
                            half  *a, 
                            half  *b,
                            float *c,
                            float *d,
                            int tid){
    uint32_t *ia0 =(uint32_t*)a;
    
    uint32_t *ib0 = (uint32_t*)b;
   
    float* c0 = c;
    float* d0 = d;
    

    // int tid = blockDim.x*blockIdx.x+threadIdx.x; // wrap: 32Thread.
    int row = tid / 4;
    int col = tid % 4;

    // int ab_index = row*4+col;
    int a0_index = row*8+col;
    int a1_index = 64+row*8+col;
    int a2_index = row*8+col+4;
    int a3_index = 64+row*8+col+4;

    int b0_index = row*8+col;
    int b1_index = row*8+col+4;

    int cd0_index = row*8+col*2;
    int cd1_index = 64+row*8+col*2;
  

    asm volatile(
     "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
     "  {%0,%1,%2,%3},  "
     "  {%4,%5,%6,%7}, "
     "  {%8,%9}, "
     "  {%10,%11,%12,%13};  "
     :
     "=f"(d0[cd0_index]),"=f"(d0[cd0_index+1]),"=f"(d0[cd1_index]),"=f"(d0[cd1_index+1])
     :
     "r"(ia0[a0_index]),"r"(ia0[a1_index]),"r"(ia0[a2_index]),"r"(ia0[a3_index]),
     "r"(ib0[b0_index]),"r"(ib0[b1_index]),
     "f"(c0[cd0_index]),"f"(c0[cd0_index+1]),"f"(c0[cd1_index]),"f"(c0[cd1_index+1])
    );
}
#endif