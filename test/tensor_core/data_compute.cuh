#ifndef __DATA_COMPUTE_H__
#define __DATA_COMPUTE_H__

__device__ void mma_m16n8k16_fp16fp32_v2(
                            half  *a, 
                            half  *b,
                            float *c,
                            float *d,
                            float *e,
                            float *f,
                            int tid){
    uint32_t *ia0 = (uint32_t*)a;
    // int tid = blockDim.x*blockIdx.x+threadIdx.x; // wrap: 32Thread.
    int a0_index = tid;
    int a1_index = tid+32;
    int a2_index = tid+64;
    int a3_index = tid+96;

    unsigned addr = (unsigned)__cvta_generic_to_shared(&b[tid*8]);
    uint32_t RC[4];
    asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(RC[0]), "=r"(RC[1]),"=r"(RC[2]), "=r"(RC[3]) : "r"(addr));

    asm volatile(
     "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
     "  {%0,%1,%2,%3},  "
     "  {%4,%5,%6,%7}, "
     "  {%8,%9}, "
     "  {%10,%11,%12,%13};  "
     :
     "=f"(d[0]),"=f"(d[1]),"=f"(d[2]),"=f"(d[3])
     :
     "r"(ia0[a0_index]),"r"(ia0[a1_index]),"r"(ia0[a2_index]),"r"(ia0[a3_index]),
    //  "r"(ib0[b0_index]),"r"(ib0[b1_index]),
     "r"(RC[0]),"r"(RC[1]),
    //  "r"(((uint32_t*)&b0[0])[0]),"r"(((uint32_t*)&b1[0])[0]),
     "f"(c[0]),"f"(c[1]),"f"(c[2]),"f"(c[3])
    );

     asm volatile(
     "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
     "  {%0,%1,%2,%3},  "
     "  {%4,%5,%6,%7}, "
     "  {%8,%9}, "
     "  {%10,%11,%12,%13};  "
     :
     "=f"(f[0]),"=f"(f[1]),"=f"(f[2]),"=f"(f[3])
     :
     "r"(ia0[a0_index]),"r"(ia0[a1_index]),"r"(ia0[a2_index]),"r"(ia0[a3_index]),
    //  "r"(ib0[b0_index]),"r"(ib0[b1_index]),
     "r"(RC[2]),"r"(RC[3]),
    //  "r"(((uint32_t*)&b0[0])[0]),"r"(((uint32_t*)&b1[0])[0]),
     "f"(e[0]),"f"(e[1]),"f"(e[2]),"f"(e[3])
    );
}


__device__ void mma_m16n8k16_fp16fp32_v3(
                            uint32_t  *RA, 
                            uint32_t  *RB,
                            float *c,
                            float *d,
                            float *e,
                            float *f){
    asm volatile(
     "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
     "  {%0,%1,%2,%3},  "
     "  {%4,%5,%6,%7}, "
     "  {%8,%9}, "
     "  {%10,%11,%12,%13};  "
     :
     "=f"(d[0]),"=f"(d[1]),"=f"(d[2]),"=f"(d[3])
     :
     "r"(RA[0]),"r"(RA[1]),"r"(RA[2]),"r"(RA[3]),
    //  "r"(ib0[b0_index]),"r"(ib0[b1_index]),
     "r"(RB[0]),"r"(RB[1]),
    //  "r"(((uint32_t*)&b0[0])[0]),"r"(((uint32_t*)&b1[0])[0]),
     "f"(c[0]),"f"(c[1]),"f"(c[2]),"f"(c[3])
    );

     asm volatile(
     "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
     "  {%0,%1,%2,%3},  "
     "  {%4,%5,%6,%7}, "
     "  {%8,%9}, "
     "  {%10,%11,%12,%13};  "
     :
     "=f"(f[0]),"=f"(f[1]),"=f"(f[2]),"=f"(f[3])
     :
     "r"(RA[0]),"r"(RA[1]),"r"(RA[2]),"r"(RA[3]),
    //  "r"(ib0[b0_index]),"r"(ib0[b1_index]),
     "r"(RB[2]),"r"(RB[3]),
    //  "r"(((uint32_t*)&b0[0])[0]),"r"(((uint32_t*)&b1[0])[0]),
     "f"(e[0]),"f"(e[1]),"f"(e[2]),"f"(e[3])
    );
}
#endif