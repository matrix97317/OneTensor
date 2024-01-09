#ifndef __GEMM_API_H__
#define __GEMM_API_H__
#include <cute/tensor.hpp>


inline __device__ __host__ size_t div_ceil(size_t a, size_t b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

namespace ConfigAttr{
using namespace cute;
template<typename T, int TileM = 128, int TileN = 128, int KStage=3>
struct GemmConfig{
    using DT = T;
    // Define k-stage
    static constexpr int K_STAGE=KStage;
    // Define Threading Block Size (TB)
    static constexpr int TB_M=TileM;
    static constexpr int TB_N=TileN;
    static constexpr int TB_K = 32;

    // Define MMA (1 Warp)
    using mma_op = SM80_16x8x16_F16F16F16F16_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    // Define Warp Block Size (WB) // 4 warp
    static constexpr int WB_M = 32;
    static constexpr int WB_N = 32;
    static constexpr int WB_K = 16;

    static constexpr int thread_repeat_M = 2;
    static constexpr int thread_repeat_N = 2;
    static constexpr int thread_repeat_K = 1;

    static constexpr int val_repeat_M = 1;
    static constexpr int val_repeat_N = 2;
    static constexpr int val_repeat_K = 1;

    using mma_thread_repeat = decltype(make_layout(make_shape(Int<thread_repeat_M>{},
                                                     Int<thread_repeat_N>{},
                                                     Int<thread_repeat_K>{}
                                                    )));
    using mma_val_repeat = decltype(make_layout(make_shape(Int<val_repeat_M>{},
                                                     Int<val_repeat_N>{},
                                                     Int<val_repeat_K>{}
                                                    )));
    // WarpMMA consist of mma
    using warp_mma = decltype (make_tiled_mma(mma_atom{},
                                mma_thread_repeat{},
                                mma_val_repeat{}));

    // Define Copy: Gemm2Smem
    using copy_gmem2smem_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>; 
    // copy 16 byte for once; For type half, have 8 elements.
    static constexpr int GROUPS_IN_ROW = TB_K / 8;
    using copy_gmem2smem_traits = Copy_Traits<copy_gmem2smem_op>;
    using copy_gmem2smem_atom = Copy_Atom<copy_gmem2smem_traits, DT>;
    // Define Copy Thread Layout / Value Layout
    // (ThreadID,ValueID) -> Offset
    using copy_gmem2smem_A = decltype(make_tiled_copy(
            copy_gmem2smem_atom{},
            make_layout(make_shape(Int<WB_M>{},Int<GROUPS_IN_ROW>{}),
                        make_stride(Int<GROUPS_IN_ROW>{},Int<1>{})),
            make_layout(make_shape(Int<1>{},Int<8>{}))
    ));
    using copy_gmem2smem_B = copy_gmem2smem_A;


    // Define Share Memory Layout:
    static constexpr int ShmLoadSwizzleM = 3; // 2^3 elemenst in package data.
    static constexpr int ShmLoadSwizzleS = 3; // 2^3 the number of package data in row;
    static constexpr int ShmLoadSwizzleB = 3; // 2^3 the number of row;

    using smem_layout_atom = decltype(composition(
        Swizzle<ShmLoadSwizzleB,ShmLoadSwizzleM,ShmLoadSwizzleS>{},
        make_layout(make_shape(Int<8>{},Int<TB_K>{}),make_stride(Int<TB_K>{},Int<1>{}))
    ));
    using smem_layout_A = decltype(tile_to_shape(
        smem_layout_atom{},
        make_shape(Int<TB_M>(),Int<TB_K>(),Int<K_STAGE>{})
    ));
    using smem_layout_B = decltype(tile_to_shape(
        smem_layout_atom{},
        make_shape(Int<TB_N>(),Int<TB_K>(),Int<K_STAGE>{})
    ));
    // Define Copy: Smem2Register
    using copy_smem2reg_op = SM75_U32x4_LDSM_N;
    using copy_smem2reg_traits = Copy_Traits<copy_smem2reg_op>;
    using copy_smem2reg_atom = Copy_Atom<copy_smem2reg_traits,DT>;

    using copy_smem2reg_A = copy_smem2reg_atom;
    using copy_smem2reg_B = copy_smem2reg_atom;
    

    //---- Define C Share Memory Layout
    using smem_layout_C_atom = decltype(composition(
        Swizzle<2,3,3>{},
        make_layout(make_shape(Int<WB_M>{},Int<WB_N>{}),make_stride(Int<WB_N>{},Int<1>{}))
    ));
    using smem_layout_C = decltype(tile_to_shape(
        smem_layout_C_atom{},
        make_shape(Int<WB_M>(),Int<WB_N>(),Int<2>{})
    ));
    //---- Define C reg2smem
    using copy_C_reg2smem = Copy_Atom<UniversalCopy<int>,DT>;
    //---- Define C smem2gmem
    using copy_C_smem2gmem_atom = Copy_Atom<UniversalCopy<cute::uint128_t>,DT>;
    using copy_C_smem2gmem = decltype(
        make_tiled_copy(copy_C_smem2gmem_atom{},
            make_layout(make_shape(Int<WB_M>{},Int<GROUPS_IN_ROW>{}),
                        make_stride(Int<GROUPS_IN_ROW>{},Int<1>{})),
            make_layout(make_shape(Int<1>{},Int<8>{}))
        )
    );
    // other info
    static constexpr int block_thread_size = size(warp_mma{});
    static constexpr int smem_ab_size = cute::cosize(smem_layout_A{})+cute::cosize(smem_layout_B{});
    static constexpr int smem_c_size = cute::cosize(smem_layout_C{});
    static constexpr int smem_size = cute::max(smem_ab_size,smem_c_size)*sizeof(DT);
};
}
template<typename DT, typename CT, typename Config>
__global__ void cute_gemm_kernel(DT* a, 
                                 DT* b,
                                 CT* c,
                                 int m,int n,int k){
    using namespace cute;
    constexpr int TB_M = Config::TB_M;
    constexpr int TB_N = Config::TB_N;
    constexpr int TB_K = Config::TB_K;
    int tid = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    // Create Logic Tensor Expression
    Tensor A = make_tensor(make_gmem_ptr((DT*)a), 
                        make_shape(m,k),
                        make_stride(k,Int<1>{})); //Shape (M,K)
    
    Tensor B = make_tensor(make_gmem_ptr((DT*)b), 
                        make_shape(n,k),
                        make_stride(k,Int<1>{})); //Shape (N,K)

    Tensor C = make_tensor(make_gmem_ptr((CT*)c), 
                        make_shape(m,n),
                        make_stride(n,Int<1>{})); //Shape (M,N)
    // Create Thread Block
    
    Tensor gmem_A = local_tile(A,make_tile(Int<TB_M>{},
                                           Int<TB_K>{}),
                                 make_coord(by,_)); //Shape((TB_M,TB_K),K/TB_K)）

    Tensor gmem_B = local_tile(B,make_tile(Int<TB_N>{},
                                           Int<TB_K>{}),
                                 make_coord(bx,_)); //Shape((TB_N,TB_K),K/TB_K)）

    Tensor gmem_C = local_tile(C,make_tile(Int<TB_M>{},
                                            Int<TB_N>{}),
                                 make_coord(by,bx)); //Shape((TB_M,TB_N)）


    // // Create Share Memory
    using smem_layout_A = typename Config::smem_layout_A;
    using smem_layout_B = typename Config::smem_layout_B;
    extern __shared__ DT smem_data[];
    DT * smem_data_A = smem_data;
    DT * smem_data_B = smem_data+cute::cosize(smem_layout_A{});
    Tensor smem_A = make_tensor(make_smem_ptr(smem_data_A),smem_layout_A{});
    //Shape (TB_M,TB_K,K_STAGE)
    Tensor smem_B = make_tensor(make_smem_ptr(smem_data_B),smem_layout_B{});
    //Shape (TB_N,TB_K,K_STAGE)

    // Create Register
    using warp_mma = typename Config::warp_mma;
    warp_mma warp_wise_mma;
    auto thread_mma = warp_wise_mma.get_slice(tid);
    auto reg_A = thread_mma.partition_fragment_A(gmem_A(_,_,0));
    //Shape ((WB_M,WB_K),TB_M/WB_M, TB_K/WB_K)
    auto reg_B = thread_mma.partition_fragment_B(gmem_B(_,_,0));
    //Shape ((WB_N,WB_K),TB_N/WB_N, TB_K/WB_K)
    auto reg_C = thread_mma.partition_fragment_C(gmem_C(_,_));
    //Shape ((WB_M,WB_N),TB_M/WB_M, TB_N/WB_N)
    clear(reg_C);
    // Create Copy Tensor (Gmem 2 Smem)
    using copy_gmem2smem_A = typename Config::copy_gmem2smem_A;
    copy_gmem2smem_A gmem2smem_copyer_A;
    auto gmem2smem_thr_copyer_A = gmem2smem_copyer_A.get_slice(tid);
    auto gmem_src_tensor_A = gmem2smem_thr_copyer_A.partition_S(gmem_A);
    //Shape ((32,32), TB_M/32, TB_K/32, K/TB_K)
    auto smem_dst_tensor_A = gmem2smem_thr_copyer_A.partition_D(smem_A);
    //Shape ((32,32), TB_M/32, TB_K/32, K_STAGE)
    using copy_gmem2smem_B = typename Config::copy_gmem2smem_B;
    copy_gmem2smem_B gmem2smem_copyer_B;
    auto gmem2smem_thr_copyer_B = gmem2smem_copyer_B.get_slice(tid);
    auto gmem_src_tensor_B = gmem2smem_thr_copyer_B.partition_S(gmem_B);
    //Shape ((32,32), TB_N/32, TB_K/32, K/TB_K)
    auto smem_dst_tensor_B = gmem2smem_thr_copyer_B.partition_D(smem_B);
    //Shape ((32,32), TB_N/32, TB_K/32, K_STAGE)

    // Create Copy Tensor (Smem 2 Reg)
    using copy_smem2reg_A = typename Config::copy_smem2reg_A;
    auto smem2reg_copyer_A = make_tiled_copy_A(copy_smem2reg_A{}, warp_wise_mma);
    auto smem2reg_thr_copyer_A = smem2reg_copyer_A.get_slice(tid);
    auto smem_src_tensor_A = smem2reg_thr_copyer_A.partition_S(smem_A);
    //Shape((WB_M,WB_K)), TB_M/WB_M, TB_K/WB_K, K_STAGE)
    auto reg_dst_tensor_A = smem2reg_thr_copyer_A.retile_D(reg_A);
    //Shape((WB_M,WB_K)), TB_M/WB_M, TB_K/WB_K)
    using copy_smem2reg_B = typename Config::copy_smem2reg_B;
    auto smem2reg_copyer_B = make_tiled_copy_B(copy_smem2reg_B{}, warp_wise_mma);
    auto smem2reg_thr_copyer_B = smem2reg_copyer_B.get_slice(tid);
    auto smem_src_tensor_B = smem2reg_thr_copyer_B.partition_S(smem_B);
    //Shape((WB_N,WB_K)), TB_N/WB_N, TB_K/WB_K, K_STAGE)
    auto reg_dst_tensor_B = smem2reg_thr_copyer_B.retile_D(reg_B);
    //Shape((WB_N,WB_K)), TB_N/WB_N, TB_K/WB_K)

    // //------ Compute --------

    // // prefech data: gmem to smem
    int tile_k_read = 0;
    int tile_k_smem_write = 0;
    int tile_k_smem_read = 0;
    constexpr int KSATGE = Config::K_STAGE;
    for(int istage=0; istage<(KSATGE-1); istage++){
        cute::copy(gmem2smem_copyer_A, gmem_src_tensor_A(_,_,_, istage),
                                        smem_dst_tensor_A(_,_,_, istage));
        cute::copy(gmem2smem_copyer_B, gmem_src_tensor_B(_,_,_,istage),
                                        smem_dst_tensor_B(_,_,_,istage));
        cp_async_fence();
        tile_k_read++;
        tile_k_smem_write++;
    }
    cp_async_wait<KSATGE-2>();
    __syncthreads();

    // prefech data: smem to reg
    int ik = 0;
    cute::copy(smem2reg_copyer_A, smem_src_tensor_A(_,_,ik,tile_k_smem_read),reg_dst_tensor_A(_,_,ik));
    cute::copy(smem2reg_copyer_B, smem_src_tensor_B(_,_,ik,tile_k_smem_read),reg_dst_tensor_B(_,_,ik));


    int k_tiles = k / Config::TB_K;
    for(int outlier_k = 0; outlier_k<k_tiles; outlier_k++){
        int inner_k_tiles = size<2>(reg_A);
        for(ik=0;ik < inner_k_tiles; ik++){
            // smem to reg (next)
            int ik_next = (ik+1) % inner_k_tiles;
            if (ik==inner_k_tiles-1){
                cp_async_wait<KSATGE-2>();
                __syncthreads();
                tile_k_smem_read = (tile_k_smem_read+1) % KSATGE;
            }
            cute::copy(smem2reg_copyer_A, smem_src_tensor_A(_,_,ik_next,tile_k_smem_read),reg_dst_tensor_A(_,_,ik_next));
            cute::copy(smem2reg_copyer_B, smem_src_tensor_B(_,_,ik_next,tile_k_smem_read),reg_dst_tensor_B(_,_,ik_next));
            // gmem to smem (next)
            if(ik==0){
                if(tile_k_read < k_tiles){
                    cute::copy(gmem2smem_copyer_A, gmem_src_tensor_A(_,_,_, tile_k_read),
                                            smem_dst_tensor_A(_,_,_, tile_k_smem_write));
                    cute::copy(gmem2smem_copyer_B, gmem_src_tensor_B(_,_,_,tile_k_read),
                                                    smem_dst_tensor_B(_,_,_,tile_k_smem_write));
                    tile_k_read++;
                    tile_k_smem_write = (tile_k_smem_write+1) % KSATGE;
                }
                cp_async_fence();
            }
            // Compute ik 
            cute::gemm(warp_wise_mma,reg_C,reg_A(_,_,ik),reg_B(_,_,ik),reg_C);            
        }
    }
    // C: reg -> smem -> gmem
    // Create C Share memory
    using smem_layout_C = typename Config::smem_layout_C;
    auto smem_C = make_tensor(smem_A(_,_,tile_k_smem_read).data(),smem_layout_C{});
    // Shape (WM,WN,2)
    using copy_C_reg2smem = typename Config::copy_C_reg2smem;
    auto reg2smem_copyer_C = make_tiled_copy_C(copy_C_reg2smem{}, warp_wise_mma);
    auto reg2smem_thr_copyer_C = reg2smem_copyer_C.get_slice(tid);
    auto reg_src_tensor_C = reg2smem_thr_copyer_C.retile_S(reg_C);
    //Shape ((WB_M,WB_N),TB_M/WB_M, TB_N/WB_N)
    auto smem_dst_tensor_C = reg2smem_thr_copyer_C.partition_D(smem_C);
    //Shape((WB_N,WB_K)),1, 1 ,2)
    using copy_C_smem2gmem = typename  Config::copy_C_smem2gmem;
    copy_C_smem2gmem smem2gmem_coper_C;
    auto smem2gmem_thr_copyer_C = smem2gmem_coper_C.get_thread_slice(tid);
    auto smem_src_tensor_C = smem2gmem_thr_copyer_C.partition_S(smem_C);
    //Shape((WB_N,WB_K)),1, 1 ,2)
    auto gmem_dst_tensor_C = smem2gmem_thr_copyer_C.partition_D(gmem_C);
    //Shape ((32,32), TB_M/32, TB_N/32)
    auto step = size<3>(smem_dst_tensor_C);
    auto reg_src_tensor_C_linear = group_modes<1,3>(reg_src_tensor_C);
    //Shape ((WB_M,WB_N),(TB_M/WB_M)*(TB_N/WB_N))
    auto gmem_dst_tensor_C_linear = group_modes<1,3>(gmem_dst_tensor_C);
    //Shape ((32,32), (TB_M/32) * (TB_N/32))

    for(int i=0; i<size<1>(gmem_dst_tensor_C_linear);i+=step){
        for(int j=0; j<step; j++){
            auto t = make_tensor_like<DT>(reg_src_tensor_C_linear(_,i+j));
            cute::copy(reg_src_tensor_C_linear(_,i+j), t);
            cute::copy(reg2smem_copyer_C, t, smem_dst_tensor_C(_,0,0,j));
        }
        __syncthreads();
        for(int j=0; j<step; j++){
            cute::copy(smem2gmem_coper_C, smem_src_tensor_C(_,0,0,j), gmem_dst_tensor_C_linear(_, i+j));
        }
        __syncthreads();
    }
}
#endif