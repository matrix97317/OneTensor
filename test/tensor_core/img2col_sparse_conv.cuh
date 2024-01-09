#ifndef __SPARSE_CONV_H__
#define __SPARSE_CONV_H__

#define FLOAT4(ptr) (reinterpret_cast<float4*>(ptr))
#define UINT(ptr) (reinterpret_cast<uint32_t*>(ptr))

texture<int, 1> texref;

template<typename DT>
__global__ void img2col(DT* inputs_tensor, DT* i2c_inputs_tensor,int N, int channel_stride ){
    // inputs_tensor: N,H,W,IC
    // i2c_inputs_tensor: N-OH-OW, KH-KW-IC
    int tid = blockIdx.x * blockDim.x+threadIdx.x;
    if(tid<N){
        int offset=tex1Dfetch(texref, tid*channel_stride);
        if(offset==-1){
            i2c_inputs_tensor[tid*channel_stride]=0;
        }else{
            // FLOAT4(&i2c_inputs_tensor[tid*channel_stride])[0] = FLOAT4(&inputs_tensor[offset])[0];
            i2c_inputs_tensor[tid]=inputs_tensor[offset];
        }
    }
}
inline __device__ __host__ size_t div_ceil(size_t a, size_t b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

void get_oh_ow(int* oh, int* ow,int h,int w, int kh, int kw, int pad_h,int pad_w,int s_h,int s_w){
    *oh = ((h+2*pad_h-kh) / (s_h)) +1;
    *ow = ((w+2*pad_w-kw) / (s_w)) +1;
}
void pre_comp(int* offset_vec,
            int bs, 
            int oh, 
            int ow, 
            int kh,
            int kw,
            int ic, 
            int h, 
            int w,
            int pad_h,
            int pad_w,
            int s_h,
            int s_w ){
    int cnt=0;
    for(int i=0;i<bs*oh*ow;i++){
        int n = i / (oh*ow);
        int npq_residual = i % (oh*ow);
        int row = npq_residual / ow;
        int col = npq_residual % ow;
        for(int j=0;j<kh*kw*ic;j++){
            int c = j % ic; //(kh*kw);
            int crs_residual = j / ic; // (kh*kw);
            int r = crs_residual / kw;
            int s = crs_residual % kw;

            int ih = row*s_h-pad_h+r;
            int iw = col*s_w-pad_w+s;           
    
            if ((ih <0) || (ih >= h) || (iw <0) || (iw >= w)){
                offset_vec[cnt]=-1;
            }else{
                int offset = n*h*w*ic+ih*w*ic+iw*ic+c;
                offset_vec[cnt]=offset;
            }
            cnt++;
        }
    }

}
// void sparse_conv2d(half* inputs_tensor,
//                    half*i2c_inputs_tensor,
//                    half* weights_tensor,
//                    half*outputs_tensor){
    
// }
#endif