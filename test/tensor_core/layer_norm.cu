#include <one_tensor.h>
#include <cute/tensor.hpp>


inline __device__ __host__ size_t div_ceil(size_t a, size_t b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}
namespace ConfigAttr{
using namespace cute;
template<typename DT,int BROW=4,int BDIM=128,int WARP_SIZE=32>
struct LNConfig{
    static constexpr int kThreadNum = BROW*WARP_SIZE;
    static constexpr int kWarpSize = WARP_SIZE;
    static constexpr int kBrow = BROW;
    static constexpr int kBdim = BDIM;
    //Define Share Memory Layout
    static constexpr int ShmSwizzleB=2;
    static constexpr int ShmSwizzleM=2;
    static constexpr int ShmSwizzlwS=3;
    using smem_layout_atom = decltype(composition(Swizzle<ShmSwizzleB,ShmSwizzleM,ShmSwizzlwS>{},make_layout(make_shape(Int<kBrow>{},Int<kWarpSize>{}),make_stride(Int<kWarpSize>{},Int<1>{})
                                            )));
    using smem_layout = decltype(tile_to_shape(smem_layout_atom{},make_shape(Int<kBrow>{},Int<kBdim>{})));
    static constexpr int shm_size = cute::cosize(smem_layout{})*sizeof(DT);
    //Define Copy: gmem to smem
    using copy_gmem2smem_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using copy_gmem2smem_triats = Copy_Traits<copy_gmem2smem_op>;
    using copy_gmem2smem_atom = Copy_Atom<copy_gmem2smem_triats,DT>;
    static constexpr int elements_per_thread = BDIM/WARP_SIZE;
    using copy_gmem2smem = decltype(make_tiled_copy(copy_gmem2smem_atom{},
                            make_layout(make_shape(Int<kBrow>{},Int<WARP_SIZE>{}),
                                        make_stride(Int<WARP_SIZE>{},Int<1>{})),
                            make_layout(make_shape(Int<1>{},Int<elements_per_thread>{}))));
    
    //Define Copy: smem to reg
    using copy_smem2gmem_atom = Copy_Atom<UniversalCopy<cute::uint32_t>,DT>;
    using copy_smem2gmem = decltype(make_tiled_copy(copy_smem2gmem_atom{},
                            make_layout(make_shape(Int<kBrow>{},Int<kWarpSize>{}),
                                        make_stride(Int<kWarpSize>{},Int<1>{})),
                            make_layout(make_shape(Int<1>{},Int<1>{}))));
    
    using reg_layout = decltype(make_layout(make_shape(Int<kBrow>{},Int<kWarpSize>{}),make_stride(Int<kWarpSize>{},Int<1>{})));

};
}

template<typename DT, typename Config>
__global__ void layer_norm(DT* a,DT* out,int rows,int dims,DT alpha, DT beta, DT eps){
    using namespace cute;
    int tid = threadIdx.x;
    int bx = blockIdx.x;
    Tensor A = make_tensor(make_gmem_ptr((DT*)a),
                           make_shape(rows,dims),
                           make_stride(dims,Int<1>{}));
                           // (rows,dims)
    Tensor Out = make_tensor(make_gmem_ptr((DT*)out),
                           make_shape(rows,dims),
                           make_stride(dims,Int<1>{}));
    constexpr int b_rows = Config::kBrow;
    constexpr int b_dims = Config::kBdim;

    Tensor gmem_a = local_tile(A,make_tile(Int<b_rows>{},Int<b_dims>{}),make_coord(bx,_));
    // shape (b_rows,b_dims,b_dims/b_dims)
    Tensor gmem_out = local_tile(Out,make_tile(Int<b_rows>{},Int<b_dims>{}),make_coord(bx,_));
    // shape (b_rows,b_dims,b_dims/b_dims)
    
    // Gmem to Smem
    extern __shared__ DT smem_data[];
    DT* smem_data_a = smem_data;
    using smem_layout = typename Config::smem_layout;
    Tensor smem_a = make_tensor(make_smem_ptr(smem_data_a),smem_layout{});
    // shape (b_rows,b_dims)
    using copy_gmem2smem = typename Config::copy_gmem2smem;
    copy_gmem2smem gmem2smem_copyer;
    auto gmem2smem_thr_copyer = gmem2smem_copyer.get_slice(tid);
    auto gmem_src_tensor = gmem2smem_thr_copyer.partition_S(gmem_a(_,_,0));
    // shape ((b_rows,b_dims),1,1)
    auto smem_dst_tensor = gmem2smem_thr_copyer.partition_D(smem_a);
    // shape ((b_rows,b_dims),1,1)
   
    // Smem to reg
    using reg_layout = typename Config::reg_layout;
    auto reg_tensor = make_tensor_like<DT>(reg_layout{});
    // shape (4,32)
    using copy_smem2reg = typename Config::copy_smem2gmem;
    copy_smem2reg  smem2reg_copyer;
    auto smem2reg_thr_copyer = smem2reg_copyer.get_slice(tid);
    auto smem_src_tensor = smem2reg_thr_copyer.partition_S(smem_a);
    // shape((4,32),1,b-dims/32)
    auto reg_dst_tensor = smem2reg_thr_copyer.partition_D(reg_tensor);
    // shape((4,32),1,1)
   
    auto reg_src_tensor = local_partition(gmem_out(_,_,0),reg_layout{},tid);
    // shape(b_rows/reg_rows,b_dims/reg_dims)
    
    // Gmem to Smem
    cute::copy(gmem2smem_copyer,gmem_src_tensor(_,_,_),smem_dst_tensor(_,_,_));
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();
    // Smem to reg
    // welford algorithm
    DT mean = 0;
    DT var = 0;
    int count = 0 ;
   
    for(int i=0; i<size<2>(smem_src_tensor);i++){
        cute::copy(smem2reg_copyer,smem_src_tensor(_,0,i),reg_dst_tensor(_,0,0));
        count = count+1;
        auto delta = reg_dst_tensor(tid,0,0)-mean;
        mean = mean+delta/count;
        auto delta2= reg_dst_tensor(tid,0,0)-mean;
        var = var+delta*delta2;
    }
    // Merge
    for(int offset=16; offset>0; offset=offset/2 ){
        auto tmp_count = __shfl_down_sync(0xffffffff,count,offset);
        auto tmp_mean = __shfl_down_sync(0xffffffff,mean,offset);
        auto tmp_var = __shfl_down_sync(0xffffffff,var,offset);
        // merge
        auto new_count = count+tmp_count;
        auto delta = mean-tmp_mean;
        auto delta2 = delta*delta;
        mean = (count * mean + tmp_count *tmp_mean) / new_count;
        var = var + tmp_var + delta2 * ( count * tmp_count) / new_count;
        count = new_count;
    }
    var = var/count;
    mean = __shfl_sync(0xffffffff,mean,0);
    var = __shfl_sync(0xffffffff,var,0);
    count = __shfl_sync(0xffffffff,count,0);
    // if(thread0()){
    //     print(count);
    //     print("\n");
    //     printf("%f\n",mean);
    //     print("\n");
    //     printf("%f\n",var);
    //     print("\n");
    // }
    __syncthreads();
    for(int i=0; i<size<2>(smem_src_tensor);i++){
        cute::copy(smem2reg_copyer,smem_src_tensor(_,0,i),reg_dst_tensor(_,0,0));
        reg_dst_tensor(tid,0,0) =  ((reg_dst_tensor(tid,0,0)-mean) / sqrt(var+eps))*alpha+beta;
        cute::copy(reg_dst_tensor(_,0,0),reg_src_tensor(_,i));
    }
    __syncthreads();
}

int main(){
    int ROW = 600;
    OneTensor<float> a(Shape(ROW,256));
    OneTensor<float> b(Shape(ROW,256));
    for(int i=0;i<a.shape.d0; i++){
        for(int j=0;j<a.shape.d1;j++){
            int index = i*a.shape.d1+j;
            a.SetHostData(j,index);
        }
    }
    // a.FillHostData(1);
    b.FillHostData(0);
    a.HostDataView();
    // b.HostDataView();
    a.sync_device();
    b.sync_device();
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    dim3 block(4*32);
    dim3 grid(div_ceil(ROW,4));
    ConfigAttr::LNConfig<float,4,256> ln_config;
    int smem_size = ln_config.shm_size;
    float alpha=1.0;
    float beta=0.0;
    float eps=0.000001;
    cudaFuncSetAttribute(layer_norm<float,decltype(ln_config)>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem_size);
    // layer_norm<float,decltype(ln_config)><<<grid,block,smem_size,stream>>>(a.deviceData<float>(),b.deviceData<float>(),ROW,256,alpha,beta,eps);
    GPU_Time((layer_norm<float,decltype(ln_config)><<<grid,block,smem_size,stream>>>(a.deviceData<float>(),b.deviceData<float>(),ROW,256,alpha,beta,eps)),
    stream,100,0);
    // b.sync_device(false);
    // b.HostDataView();
    cudaStreamDestroy(stream);
    return 0;
}