#ifndef __ONE_TENSOR_H__
#define __ONE_TENSOR_H__

#include <cuda_runtime.h>
#ifdef USE_DLPACK
#include <dlpack/dlpack.h>
#endif
#include <stdexcept>
#include <string>
#include "cuda_gadget.h"
#include <cuda_fp16.h>
struct Shape{
    size_t nDims;
    size_t d0=1,d1=1,d2=1,d3=1,d4=1;
    Shape(){};
    Shape(const Shape &s){
        this->d0=s.d0;
        this->d1=s.d1;
        this->d2=s.d2;
        this->d3=s.d3;
        this->d4=s.d4;
        this->nDims=s.nDims;
    }
    Shape(size_t D0):d0(D0){nDims=1;};
    Shape(size_t D0,size_t D1):d0(D0),d1(D1){nDims=2;};
    Shape(size_t D0,size_t D1,size_t D2):d0(D0),d1(D1),d2(D2){nDims=3;};
    Shape(size_t D0,size_t D1,size_t D2, size_t D3):d0(D0),d1(D1),d2(D2),d3(D3){nDims=4;};
    Shape(size_t D0,size_t D1,size_t D2, size_t D3,size_t D4):d0(D0),d1(D1),d2(D2),d3(D3),d4(D4){nDims=5;};

    template <typename T>
    size_t size() {
        return d0 * d1 * d2 * d3 * d4* sizeof(T);
    }
    size_t nums() {
        return d0 * d1 * d2 * d3 * d4;
    }
    size_t strides(size_t index){
        switch (index)
        {
            case 0:
                return d1 * d2 * d3 * d4;
                break;
            case 1:
                return d2 * d3 * d4;
                break;
            case 2:
                return d3 * d4;
                break;
            case 3:
                return d4;
                break;
            default:
                throw std::runtime_error("strides index error: "+std::to_string(index));
                break;
        }
    }
};
template <typename DT>
class OneTensor{
    public:
        OneTensor(Shape s){
            shape=s;
            host_data = (DT*)malloc(s.size<DT>());
        }
        ~OneTensor(){
            free(host_data);
            cudaFree(device_data);
        }
        template<typename T>
        T * deviceData(){
            if (device_data==nullptr){
                throw std::runtime_error("device_data is NULL");
            }
            return reinterpret_cast<T*>(device_data);    
        }
        template<typename T>
        T * hostData(){
            if (host_data==nullptr){
                throw std::runtime_error("host_data is NULL");
            }
            return reinterpret_cast<T*>(host_data);  
        }
        void FillHostData(DT val){
            for(size_t i=0;i<shape.nums();i++){
                host_data[i]=val;
            }
        }
        void sync_device(bool send=true){
            if (device_data==nullptr){
                cuda_checker(cudaMalloc(&device_data,shape.size<DT>()));
            }
            if(send){
                cuda_checker(cudaMemcpy(device_data,host_data,shape.size<DT>(),cudaMemcpyHostToDevice));
            }else{
                cuda_checker(cudaMemcpy(host_data,device_data,shape.size<DT>(),cudaMemcpyDeviceToHost));
            }
        }
        template<typename T>
        T GetHostData(size_t index){
            return static_cast<T>(host_data[index]);
        }
        void SetHostData(DT val,size_t index){
            host_data[index]=val;
        }
        void HostDataView(){
            if(shape.nDims !=2){
                throw std::runtime_error("Current host data view don't support nDim:"+std::to_string(shape.nDims));
            }
            for(int i=0; i<shape.d0; i++){
                for(int j=0; j<shape.d1; j++){
                    int index = i*shape.d1+j;
                    std::cout<<host_data[index]+0<<" ";
                }
                std::cout<<std::endl;
            }
            std::cout<<std::endl;
        }
        Shape shape;
    private:
        DT * host_data=nullptr;
        DT * device_data=nullptr;
};


#endif