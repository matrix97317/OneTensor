#include<one_tensor.h>
#include<cudnn.h>

#define BS 1
#define H 7
#define W 7
#define IC 2048

#define OC 2048
#define KH 3
#define KW 3

#define OH 5
#define OW 5



int main(){

    OneTensor<half> inputTensor(Shape(BS, IC, H, W));
    OneTensor<half> filterTensor(Shape(OC, IC, KH, KW));
    OneTensor<half> outputTensor(Shape(BS, OC, OH,OW));
    outputTensor.FillHostData(0);
    for(int bs=0; bs<inputTensor.shape.d0;bs++){
        for(int h=0; h<inputTensor.shape.d1; h++){
            for(int w=0; w<inputTensor.shape.d2; w++){
                for(int c=0; c<inputTensor.shape.d3; c++){
                    int index = bs*inputTensor.shape.strides(0)+\
                                h*inputTensor.shape.strides(1)+\
                                w*inputTensor.shape.strides(2)+c;
                    inputTensor.SetHostData(1+c,index);
                }
            }
        }
    }

    for(int bs=0; bs<filterTensor.shape.d0;bs++){
        for(int h=0; h<filterTensor.shape.d1; h++){
            for(int w=0; w<filterTensor.shape.d2; w++){
                for(int c=0; c<filterTensor.shape.d3; c++){
                    int index = bs*filterTensor.shape.strides(0)+\
                                h*filterTensor.shape.strides(1)+\
                                w*filterTensor.shape.strides(2)+c;
                    filterTensor.SetHostData(1+bs,index);
                }
            }
        }
    }
    
    // inputTensor.HostDataView();
    // filterTensor.HostDataView();
   

    inputTensor.sync_device();
    filterTensor.sync_device();
    outputTensor.sync_device();
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudnnHandle_t cudnn_handle;
    cudnnTensorDescriptor_t cudnnIdesc;
    cudnnFilterDescriptor_t cudnnFdesc;
    cudnnTensorDescriptor_t cudnnOdesc;
    cudnnConvolutionDescriptor_t cudnnConvDesc;
    int status=0;

    status = cudnnCreate(&cudnn_handle);
    std::cout<<"status: "<<status<<std::endl;
    status = cudnnSetStream(cudnn_handle,stream);
    std::cout<<"status: "<<status<<std::endl;
   
    status = cudnnCreateTensorDescriptor( &cudnnIdesc );
    std::cout<<"status: "<<status<<std::endl;
    status = cudnnCreateFilterDescriptor( &cudnnFdesc );
    std::cout<<"status: "<<status<<std::endl;
    status = cudnnCreateTensorDescriptor( &cudnnOdesc );
    std::cout<<"status: "<<status<<std::endl;
    status = cudnnCreateConvolutionDescriptor( &cudnnConvDesc );
    std::cout<<"status: "<<status<<std::endl;
    
    int InputsDim[4]={BS,IC,H,W};
    int InputsStride[4]={H*W*IC,H*W,W,1};
    int filterDim[4]={OC, IC, KH,KW};
    int pad[2]={2,2};
    int dilation[2]={1,1};
    int convStride[2]={2,2};
    int OutputsDim[4]={BS,OC,OH,OW};
    int OutputsStride[4]={OH*OW*OC,OH*OW, OW,1};
    
    status = cudnnSetTensorNdDescriptor(cudnnIdesc, CUDNN_DATA_HALF, 4, InputsDim , InputsStride);
    std::cout<<"status: "<<status<<std::endl;

    status = cudnnSetFilterNdDescriptor(cudnnFdesc, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, 4, filterDim);
    std::cout<<"status: "<<status<<std::endl;
    

    status = cudnnSetConvolutionNdDescriptor(cudnnConvDesc, 2, pad, convStride, dilation, CUDNN_CONVOLUTION,CUDNN_DATA_HALF);
    std::cout<<"status: "<<status<<std::endl;

    status = cudnnSetConvolutionMathType(cudnnConvDesc, CUDNN_TENSOR_OP_MATH);
    std::cout<<"status: "<<status<<std::endl;  

    status = cudnnSetTensorNdDescriptor(cudnnOdesc, CUDNN_DATA_HALF, 4, OutputsDim, OutputsStride);
    std::cout<<"status: "<<status<<std::endl;  
   
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

    void *workSpace = 0;
    size_t workSpaceSize;
    status = cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle, 
                                                     cudnnIdesc,
                                                     cudnnFdesc,
                                                     cudnnConvDesc, 
                                                     cudnnOdesc, 
                                                     algo, 
                                                     &workSpaceSize);
    std::cout<<"status: "<<status<<std::endl;
    std::cout<<"workspace size: "<<workSpaceSize<<std::endl;
    if(workSpaceSize > 0) {
        cudaMalloc(&workSpace, workSpaceSize);
    }
    float alpha=1.0;
    float beta=0.0;
 
    GPU_Time(cudnnConvolutionForward (cudnn_handle,
                            (void*)(&alpha),
                            cudnnIdesc, 
                            inputTensor.deviceData<half>(),
                            cudnnFdesc, 
                            filterTensor.deviceData<half>(),
                            cudnnConvDesc,
                            algo,
                            workSpace, 
                            workSpaceSize,
                            (void*)(&beta),
                            cudnnOdesc, 
                            outputTensor.deviceData<half>()),stream,100,0);
    
    // outputTensor.sync_device(false);
    // outputTensor.HostDataView();
    // int reshape_axis[]={0,2,3,1};
    // outputTensor.HostDataReshape(reshape_axis);
    // std::cout<<outputTensor.shape<<std::endl;
    // outputTensor.HostDataView();
    

    
    cudnnDestroyTensorDescriptor(cudnnIdesc);
    cudnnDestroyFilterDescriptor(cudnnFdesc);
    cudnnDestroyTensorDescriptor(cudnnOdesc);
    cudnnDestroyConvolutionDescriptor(cudnnConvDesc);
    cudnnDestroy(cudnn_handle);
    cudaFree(workSpace);

    return 0;
}
